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
        self.use_divide_conquer = config.get("use_divide_conquer", ["awardWonBy", "countryLandBordersCountry", "companyTradesAtStockExchange"])
        self.max_query_attempts = config.get("max_query_attempts", 3)
        
        self.temporal_strategy = config.get("temporal_strategy", "decade")

        self.count_tokens = config.get("count_tokens", False)
        
        # Logging configuration
        self.save_logs = config.get("save_logs", True)
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_suffix = f"_{self.temporal_strategy}" if self.temporal_strategy else ""
        self.prompt_log_file = self.log_dir / f"divide_conquer_prompts{strategy_suffix}_{timestamp}.jsonl"
        self.summary_log_file = self.log_dir / f"divide_conquer_summary{strategy_suffix}_{timestamp}.json"
        
        # Statistics tracking
        self.stats = {
            "total_entities": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "empty_responses": 0,
            "divide_conquer_used": 0,
            "standard_method_used": 0,
            "total_sub_queries": 0,
            "total_input_chars": 0,
            "total_output_chars": 0,
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "temporal_strategy": self.temporal_strategy,
            "start_time": datetime.now().isoformat(),
            "relations_stats": {},
            "query_details": {}
        }
        
        logger.info(f"Using temporal strategy: {self.temporal_strategy}")
        logger.info(f"Logs will be saved to: {self.prompt_log_file}")

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def log_interaction(self, entity: str, relation: str, interaction_type: str, 
                       prompt: str, response: str, processed_result: list = None, 
                       error: str = None, metadata: dict = None):
        """Log detailed information for each interaction (增加token统计)."""
        if not self.save_logs:
            return
        
        input_chars = len(prompt) if prompt else 0
        output_chars = len(response) if response else 0
        
        if self.count_tokens:
            input_tokens = self.estimate_tokens(prompt) if prompt else 0
            output_tokens = self.estimate_tokens(response) if response else 0
            
            # 更新统计
            self.stats["total_input_chars"] += input_chars
            self.stats["total_output_chars"] += output_chars
            self.stats["estimated_input_tokens"] += input_tokens
            self.stats["estimated_output_tokens"] += output_tokens
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

    def parse_entities(self, response: str, entity_type: str = "name") -> list[str]:
        """Parse different entity types from response"""
        if not response or response.lower() in ['none', 'null', '']:
            return []
        
        if entity_type in ["country", "exchange"]:
            return self.parse_general_entities(response)
        else:
            return self.parse_recipients(response)
    
    def parse_general_entities(self, response: str) -> list[str]:
        """Parse general entities (countries, exchanges) with relaxed validation"""
        entities = []
        parts = response.split(',')
        
        for part in parts:
            clean_part = part.strip()
            clean_part = re.sub(r'^\d+[\.\)]\s*', '', clean_part)
            clean_part = re.sub(r'^[-•*◦]\s*', '', clean_part).strip()
            
            if (clean_part and 
                len(clean_part) > 1 and 
                len(clean_part) < 100 and
                not re.search(r'\b(is|are|was|were|have|has)\b', clean_part.lower())):
                entities.append(clean_part)
        
        return list(set(entities))

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

    def query_by_temporal_strategy(self, award_name: str) -> set:
        if self.temporal_strategy == "year":
            return self.query_by_years(award_name)
        else:  # default to decade
            return self.query_by_decades(award_name)
    
    def query_by_years(self, award_name: str) -> set:
        all_recipients = set()
        start_year = 1950
        end_year = 2024
        
        logger.info(f"Using year-based temporal slicing for {award_name} ({start_year}-{end_year})")
        
        for year in tqdm(range(start_year, end_year + 1), desc=f"Querying years for {award_name}"):
            try:
                query = f"List all recipients of the {award_name} in {year}. Names only, no explanations. Format: Name1, Name2, Name3"
                
                self.stats["total_sub_queries"] += 1
                
                response = self.single_query_with_retry(
                    entity=award_name,
                    relation="awardWonBy",
                    prompt=query,
                    interaction_type="temporal_year",
                    metadata={
                        "temporal_type": "year",
                        "year": year
                    }
                )
                
                recipients = self.parse_recipients(response)
                if recipients:
                    logger.debug(f"  Year {year}: Found {len(recipients)} candidates")
                    all_recipients.update(recipients)
                    
            except Exception as e:
                logger.warning(f"Error querying year {year}: {e}")
                continue
        
        logger.info(f"Year-based slicing found {len(all_recipients)} total candidates")
        return all_recipients
    
    def query_by_decades(self, award_name: str) -> set:
        all_recipients = set()
        temporal_values = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
        
        logger.info(f"Using decade-based temporal slicing for {award_name}")
        
        for decade in temporal_values:
            try:
                query = f"List all recipients of the {award_name} in the {decade}. Names only, no years, no explanations. Format: Name1, Name2, Name3"
                
                self.stats["total_sub_queries"] += 1
                
                response = self.single_query_with_retry(
                    entity=award_name,
                    relation="awardWonBy", 
                    prompt=query,
                    interaction_type="temporal_decade",
                    metadata={
                        "temporal_type": "decade",
                        "decade": decade
                    }
                )
                
                recipients = self.parse_recipients(response)
                if recipients:
                    logger.debug(f"  Decade {decade}: Found {len(recipients)} candidates")
                    all_recipients.update(recipients)
                    
            except Exception as e:
                logger.warning(f"Error querying decade {decade}: {e}")
                continue
        
        logger.info(f"Decade-based slicing found {len(all_recipients)} total candidates")
        return all_recipients
    
    def query_by_dimension(self, award_name: str, dimension: str, values: list) -> set:
        """Query by specific dimension with logging."""
        all_recipients = set()
        
        for value in values:
            try:
                # Construct more precise queries
                if dimension == "nationality":
                    if value == "other":
                        query = f"List recipients of the {award_name} from non-Western countries. Names only, no countries. Format: Name1, Name2, Name3"
                    else:
                        query = f"List all {value} recipients of the {award_name}. Names only, no explanations. Format: Name1, Name2, Name3"
                else:
                    query = f"List recipients of the {award_name} related to {value}. Names only. Format: Name1, Name2, Name3"
                
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
                    logger.debug(f"  {dimension}={value}: Found {len(recipients)} valid candidates")
                    all_recipients.update(recipients)
                
            except Exception as e:
                logger.warning(f"Error querying {dimension}={value}: {e}")
                continue
        
        return all_recipients

    def comprehensive_query_by_relation(self, entity: str, relation: str) -> list:
        
        if relation == "awardWonBy":
            return self.comprehensive_award_query(entity)
        
        elif relation == "countryLandBordersCountry":
            return self.query_country_borders(entity)
        
        elif relation == "companyTradesAtStockExchange":
            return self.query_stock_exchanges(entity)
        
        else:
            return self.standard_query(entity, relation)

    def query_country_borders(self, country_name: str) -> list:
        all_borders = set()
        
        directions = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
        for direction in directions:
            query = f"Which countries border {country_name} to the {direction}? Names only."
            response = self.single_query_with_retry(country_name, "countryLandBordersCountry", 
                                                   query, f"direction_{direction}")
            borders = self.parse_entities(response, entity_type="country")
            all_borders.update(borders)
        
        query = f"List all neighboring countries of {country_name}. Names only."
        response = self.single_query_with_retry(country_name, "countryLandBordersCountry", 
                                               query, "all_neighbors")
        borders = self.parse_recipients(response)
        all_borders.update(borders)
        
        return list(all_borders)
    
    def query_stock_exchanges(self, company_name: str) -> list:
        all_exchanges = set()
        
        regions = ["American", "European", "Asian", "other international"]
        for region in regions:
            query = f"On which {region} stock exchanges does {company_name} trade? Names only."
            response = self.single_query_with_retry(company_name, "companyTradesAtStockExchange",
                                                   query, f"region_{region}")
            exchanges = self.parse_entities(response, entity_type="exchange")
            all_exchanges.update(exchanges)
        
        query = f"List all stock exchanges where {company_name} is listed. Names only."
        response = self.single_query_with_retry(company_name, "companyTradesAtStockExchange",
                                               query, "all_exchanges")
        exchanges = self.parse_recipients(response)
        all_exchanges.update(exchanges)
        
        return list(all_exchanges)
    
    def comprehensive_award_query(self, award_name: str) -> list:
        """Comprehensive divide-and-conquer query strategy."""
        logger.info(f"Starting divide-and-conquer query: {award_name}")
        
        all_recipients = set()
        
        temporal_recipients = self.query_by_temporal_strategy(award_name)
        all_recipients.update(temporal_recipients)
        
        # Geographic slicing
        geographic_values = ["American", "British", "German", "French", "Italian", 
                           "Japanese", "Canadian", "Chinese", "other"]
        geographic_recipients = self.query_by_dimension(award_name, "nationality", geographic_values)
        all_recipients.update(geographic_recipients)
        logger.info(f"Geographic slicing found {len(geographic_recipients)} candidates")
        
        # Direct enumeration
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
                    logger.info(f"Applying divide-and-conquer to {entity} ({relation})")
                    object_entities = self.comprehensive_query_by_relation(entity, relation)
                    method_used = "divide_conquer"
                else:
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
        
        if self.count_tokens:
            self.stats["total_estimated_tokens"] = (
                self.stats["estimated_input_tokens"] + 
                self.stats["estimated_output_tokens"]
            )
            self.stats["avg_tokens_per_query"] = (
                self.stats["total_estimated_tokens"] / self.stats["total_sub_queries"]
                if self.stats["total_sub_queries"] > 0 else 0
            )
        
        # Save summary
        with open(self.summary_log_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Execution summary saved to: {self.summary_log_file}")
        logger.info(f"Temporal strategy used: {self.temporal_strategy}")
        logger.info(f"Total sub-queries executed: {self.stats['total_sub_queries']}")
        
        if self.count_tokens:
            logger.info(f"Estimated total tokens: {self.stats['total_estimated_tokens']:,}")
            logger.info(f"Average tokens per query: {self.stats['avg_tokens_per_query']:.0f}")
        
def analyze_dc_performance_by_relation(self):
    for relation, stats in self.stats["relations_stats"].items():
        if stats["divide_conquer"] > 0:
            logger.info(f"\n{relation} D&C Performance:")
            logger.info(f"  - Sub-queries: {self.stats['total_sub_queries'] // stats['divide_conquer']}")
            logger.info(f"  - Success rate: {stats['success']/stats['total']:.2%}")

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
