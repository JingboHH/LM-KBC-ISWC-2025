import json
import random

from loguru import logger
from tqdm import tqdm

from models.baseline_generation_model import GenerationModel


class Qwen3Model(GenerationModel):
    def __init__(self, config):
        assert config["llm_path"] in [
            "Qwen/Qwen3-8B"
        ], (
            "The Qwen3Model class only supports the Qwen3-8B models."
        )

        super().__init__(config=config)

        self.system_message = (
            "Given a question, your task is to provide the list of answers without any other context. "
            "If there are multiple answers, separate them with a comma. "
            "If there are no answers, type \"None\".")

    def instantiate_in_context_examples(self, train_data_file):
        logger.info(f"Reading train data from `{train_data_file}`...")
        with open(train_data_file) as f:
            train_data = [json.loads(line) for line in f]

        # instantiate templates with train data
        logger.info("Instantiating in-context examples with train data...")

        in_context_examples = []
        for row in train_data:
            template = self.prompt_templates[row["Relation"]]
            example = {
                "relation": row["Relation"],
                "messages": [
                    {
                        "role": "user",
                        "content": template.format(subject_entity=row["SubjectEntity"])
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f'{", ".join(row["ObjectEntities"]) if row["ObjectEntities"] else "None"}')
                    }
                ]
            }

            in_context_examples.append(example)

        return in_context_examples

    def create_prompt(self, subject_entity: str, relation: str) -> str:
        template = self.prompt_templates[relation]
        random_examples = []
        if self.few_shot > 0:
            pool = [
                example["messages"] for example in self.in_context_examples if example["relation"] == relation
            ]
            random_examples = random.sample(
                pool,
                min(self.few_shot, len(pool))
            )

        messages = [
            {
                "role": "system",
                "content": self.system_message
            }
        ]

        for example in random_examples:
            messages.extend(example)

        messages.append({
            "role": "user",
            "content": template.format(subject_entity=subject_entity)
        })

        prompt = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_predictions(self, inputs):
        logger.info("Generating predictions...")
        prompts = [
            self.create_prompt(
                subject_entity=inp["SubjectEntity"],
                relation=inp["Relation"]
            ) for inp in inputs
        ]

        outputs = []
        for prompt in tqdm(prompts, desc="Generating predictions"):
            output = self.pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
            )
            outputs.append(output)

        logger.info("Disambiguating entities...")
        results = []
        for inp, output, prompt in tqdm(zip(inputs, outputs, prompts),
                                        total=len(inputs),
                                        desc="Disambiguating entities"):
            # remove the original prompt from the generated text
            qa_answer = output[0]["generated_text"][len(prompt):].strip()
            # parsing thinking content
            try:
                index = qa_answer.index("</think>") + len("</think>")
            except ValueError:
                index = 0
            # thinking_content = qa_answer[:index]
            qa_answer = qa_answer[index:].strip()
            if inp["Relation"] in ["hasArea", "hasCapacity"]:
                object_entities = [qa_answer]
                wikidata_ids = [qa_answer]
            else:
                object_entities = qa_answer.split(", ")
                wikidata_ids = self.disambiguate_entities(qa_answer)
            results.append({
                "SubjectEntityID": inp["SubjectEntityID"],
                "SubjectEntity": inp["SubjectEntity"],
                "Relation": inp["Relation"],
                "ObjectEntities": object_entities,
                "ObjectEntitiesID": wikidata_ids,
            })

        return results
