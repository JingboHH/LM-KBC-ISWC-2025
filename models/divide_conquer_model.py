from models.baseline_qwen_3_model import Qwen3Model
from tqdm import tqdm
import re
import json
from loguru import logger
from datetime import datetime
from pathlib import Path

class DivideConquerModel(Qwen3Model):
    """
    Enhanced Divide-and-Conquer strategy model with strict name filtering and comprehensive logging.
    
    This model specializes in high-cardinality enumeration tasks by:
    1. Decomposing queries into manageable subproblems
    2. Aggregating candidates from multiple constrained queries
    3. Applying strict validation to ensure high-quality results
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.use_divide_conquer = config.get("use_divide_conquer", ["awardWonBy"])
        self.max_query_attempts = config.get("max_query_attempts", 3)
        
        # Logging configuration
        self.save_logs = config.get("save_logs", True)
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prompt_log_file = self.log_dir / f"divide_conquer_prompts_{timestamp}.jsonl"
        self.summary_log_file = self.log_dir / f"divide_conquer_summary_{timestamp}.json"
        
        # Statistics tracking
        self.stats = {
            "total_entities": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "empty_responses": 0,
            "divide_conquer_used": 0,
            "standard_method_used": 0,
            "total_sub_queries": 0,  # Number of sub-queries in divide-conquer strategy
            "start_time": datetime.now().isoformat(),
            "relations_stats": {},
            "query_details": {}
        }
        
        logger.info(f"Divide-and-conquer strategy logs will be saved to: {self.prompt_log_file}")
    
    def log_interaction(self, entity: str, relation: str, interaction_type: str, 
                       prompt: str, response: str, processed_result: list = None, 
                       error: str = None, metadata: dict = None):
        """Log detailed information for each interaction."""
        if not self.save_logs:
            return
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "entity": entity,
            "relation": relation,
            "interaction_type": interaction_type,  # "divide_conquer", "standard", "dimension", "direct"
            "prompt": prompt,
            "raw_response": response,
            "processed_result": processed_result or [],
            "success": error is None,
            "error": error,
            "metadata": metadata or {}
        }
        
        # Write to JSONL file
        with open(self.prompt_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def update_stats(self, relation: str, success: bool, empty_result: bool = False, 
                    method_used: str = "standard"):
        """Update running statistics."""
        self.stats["total_entities"] += 1
        if success:
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1
            
        if empty_result:
            self.stats["empty_responses"] += 1
        
        if method_used == "divide_conquer":
            self.stats["divide_conquer_used"] += 1
        else:
            self.stats["standard_method_used"] += 1
            
        # Per-relation statistics
        if relation not in self.stats["relations_stats"]:
            self.stats["relations_stats"][relation] = {
                "total": 0, "success": 0, "failed": 0, "empty": 0,
                "divide_conquer": 0, "standard": 0
            }
        
        self.stats["relations_stats"][relation]["total"] += 1
        if success:
            self.stats["relations_stats"][relation]["success"] += 1
        else:
            self.stats["relations_stats"][relation]["failed"] += 1
        if empty_result:
            self.stats["relations_stats"][relation]["empty"] += 1
        if method_used == "divide_conquer":
            self.stats["relations_stats"][relation]["divide_conquer"] += 1
        else:
            self.stats["relations_stats"][relation]["standard"] += 1

    def clean_model_response(self, response: str) -> str:
        """Enhanced response cleaning function."""
        if not response:
            return "None"
            
        # Remove thinking tags
        try:
            think_end = response.index("</think>") + len("</think>")
            cleaned_response = response[think_end:].strip()
        except ValueError:
            cleaned_response = response.strip()
        
        # Remove all thinking tag remnants
        cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL)
        
        # Remove common invalid response patterns
        invalid_patterns = [
            r"I'm not certain.*?\.?\s*",
            r"I'm not sure.*?\.?\s*",
            r"I can't recall.*?\.?\s*", 
            r"I don't know.*?\.?\s*",
            r"I have to proceed.*?\.?\s*",
            r"I need to rely on.*?\.?\s*",
            r"I'll have to.*?\.?\s*",
            r"I might.*?\.?\s*",
            r"Wait\.?\s*",
            r"But\.?\s*",
            r"So\.?\s*",
            r"Let me.*?\.?\s*",
            r"I should.*?\.?\s*",
            r"I must.*?\.?\s*",
            r"Given that.*?\.?\s*",
            r"Alternatively\.?\s*",
            r"Upon checking.*?\.?\s*",
            r"After checking.*?\.?\s*"
        ]
        
        for pattern in invalid_patterns:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove excessive whitespace
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
        
        if not cleaned_response or cleaned_response.isspace():
            return "None"
            
        return cleaned_response
    
    def is_valid_name(self, text: str) -> bool:
        """Strict name validation function."""
        if not text or len(text) < 2 or len(text) > 50:  # Shortened max length
            return False
        
        text = text.strip()
        
        # Strict exclusion patterns
        invalid_patterns = [
            # Basic exclusions
            r'^(none|null|unknown|n/a|wait|but|so|and|the|a|an)',
            r'^\d+',  # Pure numbers
            r'^[^\w\s]+',  # Only punctuation

            # Exclude text containing years
            r'\b(19|20)\d{2}\b',  # Contains years
            
            # Exclude complex sentences with punctuation
            r'[?:—\-]{2,}',  # Multiple punctuation marks
            r'\?\s*(no|yes)',  # Question sentences
            
            # Exclude sentences with common words
            r'\b(was|were|is|are|from|in|then|for|not|medal|award|prize)\b',
            
            # Exclude place names and institutions
            r'\b(sweden|saudi|arabia|bangladesh|morocco|india|qatar)\b',
            r'\b(university|institute|foundation|committee|centre|center)\b',
            
            # Exclude explanatory vocabulary
            r'\b(recipient|winner|laureate|awardee|candidate)\b',
            
            # Exclude sentence structures
            r'\..*\.',  # Contains multiple periods
            r'\b(maybe|perhaps|think|again|correct|wrong)\b',
        ]
        
        text_lower = text.lower()
        for pattern in invalid_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Positive validation: strict name patterns
        # Must be capitalized word combinations
        words = text.split()
        if len(words) == 0 or len(words) > 4:  # Max 4 words
            return False
        
        # Check each word conforms to name pattern
        for word in words:
            # Each word must start with capital letter, followed by lowercase
            if not re.match(r'^[A-Z][a-z]*\.?', word):
                return False
            # Reasonable word length
            if len(word) < 2 or len(word) > 20:
                return False
        
        # At least one word length > 2 (exclude abbreviations)
        if not any(len(word.rstrip('.')) > 2 for word in words):
            return False
        
        return True
    
    def parse_recipients(self, response: str) -> list[str]:
        """Enhanced recipient parsing function with aggressive filtering."""
        if not response or response.lower() in ['none', 'null', '']:
            return []
        
        recipients = []
        
        # Preprocessing: remove obvious explanatory sentences
        # Split by periods, only keep parts that might contain names
        sentences = response.split('.')
        clean_text = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip sentences containing years or explanatory vocabulary
            if (re.search(r'\b(19|20)\d{2}\b', sentence) or 
                re.search(r'\b(was|were|then|maybe|think|correct|wrong|recipient)\b', sentence.lower())):
                continue
            clean_text += sentence + " "
        
        # If no valid content, try original text
        if not clean_text.strip():
            clean_text = response
        
        # Multiple splitting methods
        separators = [',', '\n', ';', ' and ', ' & ', '  ']  # Added double space splitting
        parts = [clean_text]
        
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Remove numbering and bullet points
            clean_part = re.sub(r'^\d+[\.\)]\s*', '', part)
            clean_part = re.sub(r'^[-•*◦]\s*', '', clean_part)
            clean_part = clean_part.strip()
            
            # Remove "and" prefix
            if clean_part.lower().startswith('and '):
                clean_part = clean_part[4:].strip()
            
            # Remove parenthetical information
            clean_part = re.sub(r'\s*\([^)]*\)', '', clean_part).strip()
            
            # Remove common suffixes
            clean_part = re.sub(r'\s+(from|in|of|the|then|was|were).*', '', clean_part, flags=re.IGNORECASE).strip()

            # Only take first possible name (before periods or other punctuation)
            clean_part = re.split(r'[.!?:]', clean_part)[0].strip()
            
            # Validate if it's a valid name
            if self.is_valid_name(clean_part):
                recipients.append(clean_part)
        
        # Deduplicate while maintaining order
        seen = set()
        unique_recipients = []
        for name in recipients:
            if name not in seen:
                seen.add(name)
                unique_recipients.append(name)
        
        return unique_recipients
    
    def create_specific_queries(self, award_name: str) -> list[str]:
        """Create more specific queries to get pure name lists."""
        return [
            f"List the names of all {award_name} recipients. Format: Name1, Name2, Name3",
            f"{award_name} winners list. Only names separated by commas.",
            f"Complete roster of {award_name} laureates. Names only.",
            f"All {award_name} recipients in chronological order. Just the names.",
            f"Who won {award_name}? List all names without years or descriptions."
        ]
    
    def single_query_with_retry(self, entity: str, relation: str, prompt: str, 
                               interaction_type: str, metadata: dict = None, max_retries: int = 2) -> str:
        """Single query with retry and logging."""
        for attempt in range(max_retries + 1):
            try:
                response = self.single_query(prompt)
                cleaned_response = self.clean_model_response(response)
                
                # Log this query
                self.log_interaction(
                    entity=entity,
                    relation=relation,
                    interaction_type=interaction_type,
                    prompt=prompt,
                    response=response,
                    processed_result=[cleaned_response] if cleaned_response and cleaned_response.lower() != 'none' else [],
                    metadata={
                        **(metadata or {}),
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "cleaned_response": cleaned_response
                    }
                )
                
                # If we get a valid response, return it
                if cleaned_response and cleaned_response.lower() not in ['none', 'null']:
                    return cleaned_response
                
                # If this is the last attempt, return result
                if attempt == max_retries:
                    return cleaned_response
                    
            except Exception as e:
                error_msg = f"Query attempt {attempt + 1} failed: {str(e)}"
                
                # Log failed query
                self.log_interaction(
                    entity=entity,
                    relation=relation,
                    interaction_type=interaction_type,
                    prompt=prompt,
                    response="",
                    error=error_msg,
                    metadata={
                        **(metadata or {}),
                        "attempt": attempt + 1,
                        "exception": str(e)
                    }
                )
                
                logger.warning(error_msg)
                if attempt == max_retries:
                    return "None"
        
        return "None"
    
    def single_query(self, prompt: str) -> str:
        """Execute single query."""
        messages = [
            {"role": "system", "content": "You are a factual assistant. Provide only the requested information without explanations, uncertainty statements, or additional context. For name lists, provide only names separated by commas."},
            {"role": "user", "content": prompt}
        ]
        
        chat_prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        output = self.pipe(
            chat_prompt, 
            max_new_tokens=min(3000, self.max_new_tokens),
            temperature=0.2,  # Further reduced temperature for more consistent results
            do_sample=True
        )
        
        generated_text = output[0]["generated_text"]
        response = generated_text[len(chat_prompt):].strip()
        
        return response
    
    def query_by_dimension(self, award_name: str, dimension: str, values: list) -> set:
        """Query by specific dimension with logging."""
        all_recipients = set()
        
        for value in values:
            try:
                # Construct more precise queries
                if dimension == "decade":
                    query = f"List all recipients of the {award_name} in the {value}. Names only, no years, no explanations. Format: Name1, Name2, Name3"
                elif dimension == "nationality":
                    if value == "other":
                        query = f"List recipients of the {award_name} from non-Western countries. Names only, no countries. Format: Name1, Name2, Name3"
                    else:
                        query = f"List all {value} recipients of the {award_name}. Names only, no explanations. Format: Name1, Name2, Name3"
                elif dimension == "field":
                    query = f"List recipients of the {award_name} for work in {value}. Names only, no descriptions. Format: Name1, Name2, Name3"
                elif dimension == "recent":
                    query = f"List recipients of the {award_name} in recent years. Names only, no years. Format: Name1, Name2, Name3"
                elif dimension == "early":
                    query = f"List early recipients of the {award_name}. Names only, no years. Format: Name1, Name2, Name3"
                else:
                    query = f"List recipients of the {award_name} related to {value}. Names only. Format: Name1, Name2, Name3"
                
                # Track sub-query statistics
                self.stats["total_sub_queries"] += 1
                
                response = self.single_query_with_retry(
                    entity=award_name,
                    relation="awardWonBy",
                    prompt=query,
                    interaction_type="dimension",
                    metadata={
                        "dimension": dimension,
                        "value": value,
                        "query_strategy": "divide_conquer"
                    }
                )
                
                recipients = self.parse_recipients(response)
                
                if recipients:
                    logger.debug(f"  {dimension}={value}: Found {len(recipients)} valid candidates: {recipients[:3]}...")
                    all_recipients.update(recipients)
                
            except Exception as e:
                logger.warning(f"Error querying {dimension}={value}: {e}")
                continue
        
        return all_recipients
    
    def comprehensive_award_query(self, award_name: str) -> list:
        """Comprehensive divide-and-conquer query strategy with logging."""
        logger.info(f"Starting enhanced divide-and-conquer query: {award_name}")
        
        all_recipients = set()
        
        # Temporal slicing: decade-based categories
        temporal_values = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
        temporal_recipients = self.query_by_dimension(award_name, "decade", temporal_values)
        all_recipients.update(temporal_recipients)
        logger.info(f"Temporal slicing found {len(temporal_recipients)} candidates")
        
        # Geographic slicing: nationality categories
        geographic_values = ["American", "British", "German", "French", "Italian", "Japanese", "Canadian", "Chinese", "other"]
        geographic_recipients = self.query_by_dimension(award_name, "nationality", geographic_values)
        all_recipients.update(geographic_recipients)
        logger.info(f"Geographic slicing found {len(geographic_recipients)} candidates")
        
        # Direct enumeration: backup strategies
        direct_queries = self.create_specific_queries(award_name)
        for i, query in enumerate(direct_queries):
            try:
                response = self.single_query_with_retry(
                    entity=award_name,
                    relation="awardWonBy",
                    prompt=query,
                    interaction_type="direct",
                    metadata={
                        "query_index": i,
                        "total_direct_queries": len(direct_queries)
                    }
                )
                
                recipients = self.parse_recipients(response)
                if recipients:
                    all_recipients.update(recipients)
                    logger.debug(f"Direct query {i+1} found {len(recipients)} candidates")
                    
            except Exception as e:
                logger.warning(f"Direct query {i+1} failed: {e}")
                continue
        
        # Convert to list and apply final validation
        final_recipients = []
        for recipient in all_recipients:
            if self.is_valid_name(recipient):
                final_recipients.append(recipient)
        
        logger.info(f"Final result for {award_name}: {len(final_recipients)} validated recipients")
        return final_recipients
    
    def generate_predictions(self, inputs):
        """Generate predictions using divide-and-conquer where appropriate."""
        logger.info("Starting enhanced divide-and-conquer strategy...")
        
        results = []
        for inp in tqdm(inputs, desc="Divide-and-conquer predictions"):
            entity = inp["SubjectEntity"]
            relation = inp["Relation"]
            
            try:
                if relation in self.use_divide_conquer:
                    # Use divide-and-conquer strategy
                    logger.info(f"Applying divide-and-conquer to {entity} ({relation})")
                    object_entities = self.comprehensive_award_query(entity)
                    method_used = "divide_conquer"
                else:
                    # Use standard method
                    object_entities = self.standard_query(entity, relation)
                    method_used = "standard"
                
                # For this challenge, we assume all entities are their own IDs
                wikidata_ids = object_entities.copy() if object_entities else []
                
                # Update statistics
                self.update_stats(relation, success=True, empty_result=len(object_entities) == 0, method_used=method_used)
                
            except Exception as e:
                logger.error(f"Error processing {entity}-{relation}: {e}")
                object_entities = []
                wikidata_ids = []
                
                # Update statistics
                self.update_stats(relation, success=False, empty_result=True, method_used="failed")
            
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
    
    def standard_query(self, entity: str, relation: str) -> list:
        """Standard query method for non-divide-conquer relations."""
        # This would implement the standard baseline approach
        # For brevity, implementing a simple version here
        prompt = f"What {relation} does {entity} have? If uncertain, answer 'none'."
        response = self.single_query_with_retry(entity, relation, prompt, "standard")
        
        if response.lower() == "none":
            return []
        else:
            return [response] if response else []
    
    def save_final_summary(self):
        """Save final execution statistics summary."""
        if not self.save_logs:
            return
            
        self.stats["end_time"] = datetime.now().isoformat()
        self.stats["duration_seconds"] = (
            datetime.fromisoformat(self.stats["end_time"]) - 
            datetime.fromisoformat(self.stats["start_time"])
        ).total_seconds()
        
        # Calculate success rates
        if self.stats["total_entities"] > 0:
            self.stats["success_rate"] = self.stats["successful_queries"] / self.stats["total_entities"]
            self.stats["empty_rate"] = self.stats["empty_responses"] / self.stats["total_entities"]
        
        # Save summary
        with open(self.summary_log_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Execution summary saved to: {self.summary_log_file}")
        logger.info(f"Success rate: {self.stats.get('success_rate', 0):.2%}")
        logger.info(f"Empty response rate: {self.stats.get('empty_rate', 0):.2%}")
        logger.info(f"Divide-conquer usage: {self.stats['divide_conquer_used']} / {self.stats['total_entities']}")
        logger.info(f"Total sub-queries executed: {self.stats['total_sub_queries']}")

# Configuration utilities
def create_divide_conquer_config():
    """Create divide-and-conquer configuration example."""
    return {
        "model_name": "Qwen/Qwen3-8B",
        "use_divide_conquer": ["awardWonBy"],
        "max_query_attempts": 3,
        "max_new_tokens": 3000,
        
        # Logging configuration
        "save_logs": True,
        "log_dir": "divide_conquer_logs",
    }