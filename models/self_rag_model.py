from models.baseline_qwen_3_model import Qwen3Model
import json
from loguru import logger
from tqdm import tqdm
import re
from datetime import datetime
from pathlib import Path
import os

class SelfRAGModel(Qwen3Model):
    """
    Self-RAG strategy implementation with comprehensive logging and token counting.
    """
    
    def __init__(self, config):
        super().__init__(config)
        # Self-RAG specific configuration
        self.use_description = config.get("use_description", True)
        self.description_max_tokens = config.get("description_max_tokens", 1000)
        
        # Token counting configuration
        self.count_tokens = config.get("count_tokens", False)
        
        # Logging configuration
        self.save_logs = config.get("save_logs", True)
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prompt_log_file = self.log_dir / f"selfrag_prompts_{timestamp}.jsonl"
        self.summary_log_file = self.log_dir / f"selfrag_summary_{timestamp}.json"
        
        # Statistics tracking with token counts
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "empty_responses": 0,
            "total_descriptions": 0,
            "total_input_chars": 0,
            "total_output_chars": 0,
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "description_tokens": 0,
            "query_tokens": 0,
            "start_time": datetime.now().isoformat(),
            "relations_stats": {}
        }
        
        logger.info(f"SelfRAG logs will be saved to: {self.prompt_log_file}")
        if self.count_tokens:
            logger.info("Token counting enabled")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens: approximately 4 characters per token for English text."""
        return len(text) // 4 if text else 0
    
    def log_interaction(self, entity: str, relation: str, interaction_type: str, 
                       prompt: str, response: str, processed_result: list = None, 
                       error: str = None, metadata: dict = None):
        """Log detailed information with token counting."""
        if not self.save_logs:
            return
        
        # Calculate character and token counts
        input_chars = len(prompt) if prompt else 0
        output_chars = len(response) if response else 0
        
        if self.count_tokens:
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(response)
            
            # Update global statistics
            self.stats["total_input_chars"] += input_chars
            self.stats["total_output_chars"] += output_chars
            self.stats["estimated_input_tokens"] += input_tokens
            self.stats["estimated_output_tokens"] += output_tokens
            
            # Track tokens by interaction type
            if interaction_type == "description":
                self.stats["description_tokens"] += input_tokens + output_tokens
            elif interaction_type == "query":
                self.stats["query_tokens"] += input_tokens + output_tokens
        else:
            input_tokens = 0
            output_tokens = 0
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "entity": entity,
            "relation": relation,
            "interaction_type": interaction_type,
            "prompt": prompt,
            "raw_response": response,
            "processed_result": processed_result or [],
            "success": error is None,
            "error": error,
            "input_chars": input_chars,
            "output_chars": output_chars,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "metadata": metadata or {}
        }
        
        # Write to JSONL file
        with open(self.prompt_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def update_stats(self, relation: str, success: bool, empty_result: bool = False, 
                    is_description: bool = False):
        """Update running statistics."""
        if is_description:
            self.stats["total_descriptions"] += 1
        else:
            self.stats["total_queries"] += 1
            if success:
                self.stats["successful_queries"] += 1
            else:
                self.stats["failed_queries"] += 1
                
            if empty_result:
                self.stats["empty_responses"] += 1
        
        # Per-relation statistics
        if relation not in self.stats["relations_stats"]:
            self.stats["relations_stats"][relation] = {
                "total": 0, "success": 0, "failed": 0, "empty": 0,
                "description_tokens": 0, "query_tokens": 0
            }
        
        if not is_description:
            self.stats["relations_stats"][relation]["total"] += 1
            if success:
                self.stats["relations_stats"][relation]["success"] += 1
            else:
                self.stats["relations_stats"][relation]["failed"] += 1
            if empty_result:
                self.stats["relations_stats"][relation]["empty"] += 1
    
    def generate_entity_description(self, entity_name: str, relation: str) -> str:
        """Stage 1: Generate entity description with token counting."""
        description_prompts = {
            "hasArea": f"Describe {entity_name} with emphasis on its total area, size measurements, and spatial dimensions in square kilometers.",
            "hasCapacity": f"Describe {entity_name} focusing on its maximum capacity, volume, or the number of people/items it can hold or accommodate.", 
            "awardWonBy": f"Describe {entity_name} focusing on who has received this award, the recipients, winners, and laureates throughout its history.",
            "countryLandBordersCountry": f"Describe {entity_name} focusing on which specific countries it shares land borders with and its neighboring nations.",
            "companyTradesAtStockExchange": f"Describe {entity_name} focusing on which stock exchanges it is listed on and where its shares are traded.",
            "personHasCityOfDeath": f"Describe {entity_name} focusing on where they died, their place of death, and the city where they passed away."
        }
        
        user_prompt = description_prompts.get(relation, f"Describe {entity_name} in detail.")
        
        messages = [
            {"role": "system", "content": "Provide a detailed, factual description."},
            {"role": "user", "content": user_prompt}
        ]
        
        chat_prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        try:
            output = self.pipe(chat_prompt, max_new_tokens=self.description_max_tokens, temperature=0.1)
            description = output[0]["generated_text"][len(chat_prompt):].strip()
            
            # Log description generation with token counts
            self.log_interaction(
                entity=entity_name,
                relation=relation,
                interaction_type="description",
                prompt=chat_prompt,
                response=description,
                metadata={
                    "max_tokens": self.description_max_tokens,
                    "temperature": 0.1
                }
            )
            
            # Update description count
            self.update_stats(relation, success=True, is_description=True)
            
            return description
            
        except Exception as e:
            error_msg = f"Description generation failed: {str(e)}"
            self.log_interaction(
                entity=entity_name,
                relation=relation,
                interaction_type="description",
                prompt=chat_prompt,
                response="",
                error=error_msg
            )
            logger.error(f"Description generation failed for {entity_name}: {e}")
            self.update_stats(relation, success=False, is_description=True)
            return f"Basic information about {entity_name}."
    
    def create_prompt_with_description(self, subject_entity: str, relation: str, description: str) -> str:
        """
        Stage 2: Create query prompt based on description.
        Uses strict output specifications to minimize post-processing.
        """
        enhanced_prompts = {
            "hasArea": f"Given this information about {subject_entity}: {description}\n\nWhat is the exact area of {subject_entity} in square kilometers? Answer with one number only.",
            
            "hasCapacity": f"Given this information about {subject_entity}: {description}\n\nWhat is the exact capacity of {subject_entity}(How many people can it accommodate)? Answer with number only.",
            
            "awardWonBy": f"Given this information about {subject_entity}: {description}\n\nWho are all the recipients of {subject_entity}? If you don't know or are uncertain about any recipients, answer 'none'. Otherwise, list names only.",
            
            "countryLandBordersCountry": f"Given this information about {subject_entity}: {description}\n\nWhich countries border {subject_entity}? If you don't know or are uncertain about the bordering countries, answer 'none'. Otherwise, list all country names only, separated by commas.",
            
            "companyTradesAtStockExchange": f"Given this information about {subject_entity}: {description}\n\nOn which stock exchanges does {subject_entity} trade? If you don't know or are uncertain, answer 'none'. Otherwise, list all exchange names without abbreviations, separated by commas.",
            
            "personHasCityOfDeath": f"Given this information about {subject_entity}: {description}\n\nIn which city did {subject_entity} die? If you don't know or are uncertain about the city, answer 'none'. Otherwise, answer with only one city name."
        }
        
        enhanced_question = enhanced_prompts.get(relation, f"Answer about {subject_entity} and {relation}. If uncertain, answer 'none'.")
        
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": enhanced_question}
        ]
        
        return self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def clean_model_response(self, raw_response: str) -> str:
        """
        Clean model response by removing thinking content and format markers.
        """
        import re
        
        # Remove complete <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        
        # Remove unclosed <think> tags
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
        
        # Remove other possible markers
        cleaned = cleaned.replace('</think>', '')
        
        # Clean excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()

    def extract_number(self, text: str) -> str:
        """Enhanced number extraction function."""
        if not text:
            return ""
        
        # Priority: match large numbers with commas (e.g., 60,206)
        comma_match = re.search(r'\b\d{1,3}(?:,\d{3})+\b', text)
        if comma_match:
            return comma_match.group().replace(',', '')
        
        # Fallback: match regular numbers
        number_match = re.search(r'\b\d+\.?\d*\b', text)
        if number_match:
            return number_match.group()
        
        return ""

    def generate_predictions(self, inputs):
        """
        Main prediction method implementing Self-RAG with comprehensive logging.
        """
        logger.info("Starting Self-RAG strategy prediction generation...")
        logger.info(f"Total queries to process: {len(inputs)}")
        
        results = []
        for inp in tqdm(inputs, desc="Self-RAG predictions"):
            entity = inp["SubjectEntity"]
            relation = inp["Relation"]
            
            try:
                if self.use_description:
                    # Stage 1: Generate description
                    description = self.generate_entity_description(entity, relation)
                    # Stage 2: Query based on description
                    prompt = self.create_prompt_with_description(entity, relation, description)
                else:
                    prompt = self.create_prompt(entity, relation)
                
                # Generate answer
                output = self.pipe(prompt, max_new_tokens=self.max_new_tokens, temperature=0.1)
                raw_answer = output[0]["generated_text"][len(prompt):].strip()
                
                # Clean response
                qa_answer = self.clean_model_response(raw_answer)
                
                # Handle empty answers
                if not qa_answer or qa_answer.lower() == "none":
                    object_entities = []
                    wikidata_ids = []
                elif relation in ["hasArea", "hasCapacity"]:
                    # Numeric relations: use enhanced number extraction
                    number_value = self.extract_number(qa_answer)
                    if number_value:
                        object_entities = [number_value]
                        wikidata_ids = [number_value]
                    else:
                        object_entities = []
                        wikidata_ids = []
                else:
                    # Other relations: split and disambiguate
                    if "," in qa_answer:
                        object_entities = [entity.strip() for entity in qa_answer.split(",")]
                    else:
                        object_entities = [qa_answer] if qa_answer else []
                    
                    # Filter out obviously non-entity content
                    object_entities = [entity for entity in object_entities 
                                    if entity and len(entity) > 1 and len(entity) < 100]
                    
                    if object_entities:
                        wikidata_ids = self.disambiguate_entities(", ".join(object_entities))
                    else:
                        wikidata_ids = []
                
                # Log query interaction
                self.log_interaction(
                    entity=entity,
                    relation=relation,
                    interaction_type="query",
                    prompt=prompt,
                    response=raw_answer,
                    processed_result=object_entities,
                    metadata={
                        "cleaned_response": qa_answer,
                        "wikidata_ids": wikidata_ids,
                        "max_tokens": self.max_new_tokens,
                        "temperature": 0.1,
                        "use_description": self.use_description
                    }
                )
                
                # Update statistics
                self.update_stats(relation, success=True, empty_result=len(object_entities) == 0)
                
            except Exception as e:
                # Log error and return empty result
                error_msg = f"Processing error: {str(e)}"
                logger.error(f"Error processing {entity}-{relation}: {e}")
                
                self.log_interaction(
                    entity=entity,
                    relation=relation,
                    interaction_type="query",
                    prompt=prompt if 'prompt' in locals() else "N/A",
                    response="",
                    error=error_msg
                )
                
                object_entities = []
                wikidata_ids = []
                
                # Update statistics
                self.update_stats(relation, success=False, empty_result=True)
            
            results.append({
                "SubjectEntityID": inp["SubjectEntityID"],
                "SubjectEntity": entity,
                "Relation": relation,
                "ObjectEntities": object_entities,
                "ObjectEntitiesID": wikidata_ids,
            })
        
        # Save final summary
        self.save_final_summary()
        
        return results
    
    def save_final_summary(self):
        """Save final execution statistics with token counts."""
        if not self.save_logs:
            return
            
        self.stats["end_time"] = datetime.now().isoformat()
        self.stats["duration_seconds"] = (
            datetime.fromisoformat(self.stats["end_time"]) - 
            datetime.fromisoformat(self.stats["start_time"])
        ).total_seconds()
        
        # Calculate success rates
        if self.stats["total_queries"] > 0:
            self.stats["success_rate"] = self.stats["successful_queries"] / self.stats["total_queries"]
            self.stats["empty_rate"] = self.stats["empty_responses"] / self.stats["total_queries"]
        
        # Token statistics
        if self.count_tokens:
            self.stats["total_estimated_tokens"] = (
                self.stats["estimated_input_tokens"] + 
                self.stats["estimated_output_tokens"]
            )
            
            # Average tokens per interaction
            total_interactions = self.stats["total_queries"] + self.stats["total_descriptions"]
            if total_interactions > 0:
                self.stats["avg_tokens_per_interaction"] = (
                    self.stats["total_estimated_tokens"] / total_interactions
                )
            
            # Token breakdown
            self.stats["token_breakdown"] = {
                "description_percentage": (
                    self.stats["description_tokens"] / self.stats["total_estimated_tokens"] * 100
                    if self.stats["total_estimated_tokens"] > 0 else 0
                ),
                "query_percentage": (
                    self.stats["query_tokens"] / self.stats["total_estimated_tokens"] * 100
                    if self.stats["total_estimated_tokens"] > 0 else 0
                )
            }
        
        # Save summary
        with open(self.summary_log_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Execution summary saved to: {self.summary_log_file}")
        logger.info(f"Success rate: {self.stats.get('success_rate', 0):.2%}")
        logger.info(f"Empty response rate: {self.stats.get('empty_rate', 0):.2%}")
        
        if self.count_tokens:
            logger.info(f"Total estimated tokens: {self.stats['total_estimated_tokens']:,}")
            logger.info(f"Description tokens: {self.stats['description_tokens']:,} ({self.stats['token_breakdown']['description_percentage']:.1f}%)")
            logger.info(f"Query tokens: {self.stats['query_tokens']:,} ({self.stats['token_breakdown']['query_percentage']:.1f}%)")

# Configuration utilities
def create_enhanced_config():
    """Create enhanced configuration example."""
    return {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "use_description": True,
        "description_max_tokens": 1000,
        "max_new_tokens": 200,
        
        # Logging configuration
        "save_logs": True,
        "log_dir": "experiment_logs",
        
        # Other configurations...
    }

# Log analysis utilities
def analyze_logs(prompt_log_file: str):
    """Analyze generated log files."""
    interactions = []
    
    with open(prompt_log_file, "r", encoding="utf-8") as f:
        for line in f:
            interactions.append(json.loads(line))
    
    print(f"Total interactions: {len(interactions)}")
    
    # Analyze by interaction type
    description_interactions = [i for i in interactions if i["interaction_type"] == "description"]
    query_interactions = [i for i in interactions if i["interaction_type"] == "query"]
    
    print(f"Description generation: {len(description_interactions)}")
    print(f"Query interactions: {len(query_interactions)}")
    
    # Analyze by relation
    relations = {}
    for interaction in interactions:
        rel = interaction["relation"]
        if rel not in relations:
            relations[rel] = {"total": 0, "success": 0, "empty": 0}
        relations[rel]["total"] += 1
        if interaction["success"]:
            relations[rel]["success"] += 1
        if not interaction["processed_result"]:
            relations[rel]["empty"] += 1
    
    print("\nStatistics by relation:")
    for rel, stats in relations.items():
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        empty_rate = stats["empty"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {rel}: success_rate={success_rate:.2%}, empty_rate={empty_rate:.2%}")
    
    return interactions
