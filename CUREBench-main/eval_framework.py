"""
Bio-Medical AI Competition Starter Kit

A simple framework for evaluating MedGemma models on bio-medical datasets.
"""

import json
import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod
import csv
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""
    dataset_name: str
    model_name: str
    accuracy: float
    correct_predictions: int
    total_examples: int
    predictions: List[Dict]
    reasoning_traces: List[str] = None
    details: Optional[Dict] = None


class MedGemmaTextModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load(self):
        print("✅ MedGemma model is already loaded in memory.")

    def inference(self, prompt: str, max_tokens: int = 256):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Structure attendue par eval_framework
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_text}]
        return response_text, messages



class CompetitionKit:
    """
    Simple competition framework for MedGemma evaluation
    """
    
    def __init__(self, config_path: str = None):
        self.model = None
        self.model_name = None
        self.config = json.load(open(config_path, 'r')) if config_path else {}
        self.output_dir = self.config.get('output_dir', 'results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.datasets = self._load_dataset_configs(self.config)
    
    def load_model(self, model_name: str, **kwargs):
        """
        Load MedGemma model for evaluation
        """
        self.model_name = model_name
        logger.info(f"Loading MedGemma model: {model_name}")
        self.model = MedGemmaModel(model_name)
        self.model.load(**kwargs)
    
    def _load_dataset_configs(self, config) -> Dict:
        """Load dataset configurations"""
        if not config:
            print("No config provided, exiting.")
            exit(1)

        if 'dataset' in config:
            dataset_config = config['dataset']
            dataset_name = dataset_config.get('dataset_name', 'treatment')
            return {dataset_name: dataset_config}
        else:
            print("No dataset config found, exiting.")
            exit(1)

    def evaluate(self, dataset_name: str) -> EvaluationResult:
        """
        Evaluate model on a dataset
        """
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")
        
        dataset_config = self.datasets[dataset_name]
        logger.info(f"Evaluating on {dataset_name}: {dataset_config['description']}")
        
        dataset = self._load_dataset(dataset_config)
        self._last_dataset_examples = dataset
        
        predictions = []
        reasoning_traces = []
        total_count = len(dataset)
        accuracy_correct_count = 0
        accuracy_total_count = 0
        
        logger.info(f"Running evaluation on {total_count} examples...")
        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                prediction, reasoning_trace = self._get_prediction_with_trace(example)
                predictions.append(prediction)
                reasoning_traces.append(reasoning_trace)
                
                question_type = example["question_type"]
                expected_answer = example.get("answer")
                
                if question_type in ["multi_choice", "open_ended_multi_choice"]:
                    if expected_answer != '':
                        is_correct = prediction["choice"] == expected_answer
                    else:
                        is_correct = False
                    accuracy_total_count += 1
                    if is_correct:
                        accuracy_correct_count += 1
                elif question_type == "open_ended":
                    if expected_answer != '':
                        is_correct = prediction["open_ended_answer"] == expected_answer
                    else:
                        is_correct = False
                
                if (i + 1) % 10 == 0:
                    current_acc = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0
                    logger.info(f"Progress: {i+1}/{total_count}, Accuracy: {current_acc:.2%}")
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                error_prediction = {
                    "choice": "NOTAVALUE",
                    "open_ended_answer": "Error"
                }
                predictions.append(error_prediction)
                reasoning_traces.append("Error occurred during inference")
        
        accuracy = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0
        
        return EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,
            total_examples=accuracy_total_count,
            predictions=predictions,
            reasoning_traces=reasoning_traces
        )

    def _load_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Load dataset based on configuration"""
        from dataset_utils import build_dataset
        from torch.utils.data import DataLoader
        
        dataset = build_dataset(dataset_config.get("dataset_path"))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_list = []
        
        for batch in dataloader:
            question_type = batch[0][0]
            
            # Base structure for all question types
            example = {
                "question_type": batch[0][0],
                "id": batch[1][0],
                "question": batch[2][0],
                "answer": batch[3][0],
            }
            
            # Dynamic options loading - handles variable number of options (A,B,C,D,E...)
            options = {}
            option_index = 4  # Start after fixed fields
            while option_index < len(batch) and len(batch[option_index]) > 0:
                option_letter = chr(65 + (option_index - 4))  # 65 = 'A' in ASCII
                options[option_letter] = batch[option_index][0]
                option_index += 1
            
            example["options"] = options
            
            # Additional field for open_ended_multi_choice
            if question_type == "open_ended_multi_choice":
                example["meta_question"] = batch[option_index][0] if option_index < len(batch) else ""
            
            dataset_list.append(example)
        
        return dataset_list

    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, str]:
        """Get model prediction and reasoning trace for a single example"""
        question = example["question"]
        question_type = example["question_type"]
        options="\n".join([f"{k}. {v}" for k, v in example["options"].items()])
        # Format prompt
        if question_type == "multi_choice":
            prompt = f"The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).\n\nQuestion: {question}\n\n Options: {options} \n\nAnswer:"
        elif question_type == "open_ended_multi_choice" or question_type == "open_ended":
            prompt = f"The following is an open-ended question about medicine. Provide a comprehensive answer.Answer with only the letter (A, B, C, D, or E).\n\nQuestion: {question}\n\n Options: {options} "
        
        # Get model response and messages using the model's inference method
        response, reasoning_trace = self.model.inference(prompt)
        
        # Initialize prediction dictionary
        prediction = {
            "choice": "",  # Use empty string instead of None
            "open_ended_answer": ""  # Use empty string instead of None
        }
        
        # Extract answer from response
        if question_type == "multi_choice":
            # For multiple choice, extract the letter
            choice = self._extract_multiple_choice_answer(response)
            # Ensure choice is never None or NULL
            prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            prediction["open_ended_answer"] = response.strip()  # Keep full response too
        elif question_type == "open_ended_multi_choice":
            # First get the detailed response
            prediction["open_ended_answer"] = response.strip()
            
            # Then use meta question to get choice, if available
            if "meta_question" in example:
                meta_prompt = f"{example['meta_question']}Agent's answer: {response.strip()}\n\nMulti-choice answer:"
                meta_response, meta_reasoning = self.model.inference(meta_prompt)
                # Combine reasoning traces
                reasoning_trace += meta_reasoning
                # Extract the letter choice
                choice = self._extract_multiple_choice_answer(meta_response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            else:
                # If no meta_question, try to extract choice directly from the response
                choice = self._extract_multiple_choice_answer(response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
        elif question_type == "open_ended":
            # For open-ended, only return response, use N/A for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE" # Use N/A instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()
        
        return prediction, reasoning_trace
   

    def _extract_multiple_choice_answer(self,response: str, valid_choices=None) -> str:
        if not response:
            return ""
        response_upper = response.upper()

        # Étape 1 : cherche les patterns prioritaires
        patterns = [
            r"FINAL ANSWER[:\s]*([ABCDE])",
            r"ANSWER[:\s]*([ABCDE])",
            r"CORRECT ANSWER IS[:\s]*([ABCDE])"
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                letter = match.group(1)
                if valid_choices and letter not in valid_choices:
                    continue  # Ignore si pas dans les options
                return letter

        # Étape 2 : dernier recours -> lettre isolée
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if letter in response_upper:
                if valid_choices and letter not in valid_choices:
                    continue
                return letter

        return ""  # Si rien trouvé
    
    def save_submission(self, results: List[EvaluationResult], filename: str = "submission.csv", 
                       metadata: Dict = None, dataset_examples: List[Dict] = None,
                       config_path: str = None, args: argparse.Namespace = None):
        """
        Save results in competition submission format
        """
        import pandas as pd
        import zipfile
        
        metadata = self.get_metadata(config_path, args, metadata)
        submission_data = []
        
        for result in results:
            examples = dataset_examples if dataset_examples else []
            
            for i, (prediction, example) in enumerate(zip(result.predictions, examples)):
                reasoning_trace = json.dumps(result.reasoning_traces[i])
                prediction_text = prediction.get("open_ended_answer", "") or ""
                
                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in ['NULL', 'NONE', 'NAN']:
                    choice_clean = "NOTAVALUE"
                elif str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"
                else:
                    choice_clean = str(choice_raw).strip()
                
                if not reasoning_trace or reasoning_trace == "null" or reasoning_trace.strip() == "":
                    reasoning_trace = "No reasoning available"
                
                row = {
                    "id": str(example.get("id", str(i)) or f"unknown_{i}"),
                    "prediction": str(prediction_text),
                    "choice": str(choice_clean),
                    "reasoning": str(reasoning_trace)
                }
                
                if str(choice_clean).upper() in ['NULL', 'NONE', 'NAN'] or str(choice_clean).strip() == "":
                    row["choice"] = "NOTAVALUE"
                
                submission_data.append(row)
        
        df = pd.DataFrame(submission_data)
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        null_replacements = {
            'id': 'unknown_id',
            'prediction': 'No prediction available',
            'choice': 'NOTAVALUE',
            'reasoning': 'No reasoning available'
        }
        
        for col in df.columns:
            df[col] = df[col].fillna(null_replacements.get(col, 'NOTAVALUE'))
            
            null_like_values = ['nan', 'NaN', 'None', 'null', 'NULL', '<NA>', 'nat', 'NaT']
            for null_val in null_like_values:
                df[col] = df[col].replace(null_val, null_replacements.get(col, 'NOTAVALUE'))
            
            if col == 'choice':
                df[col] = df[col].replace('NOTAVALUE', 'NOTAVALUE')
                for null_val in null_like_values:
                    df[col] = df[col].replace(null_val, 'NOTAVALUE')
                df[col] = df[col].replace('', 'NOTAVALUE')
                df[col] = df[col].replace(' ', 'NOTAVALUE')
            
            if col != 'choice' and col in null_replacements:
                df[col] = df[col].replace('', null_replacements[col])
                df[col] = df[col].replace(' ', null_replacements[col])
        
        csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(csv_path, index=False, na_rep='NOTAVALUE', quoting=1)
        logger.info(f"Successfully saved CSV to {csv_path}")
    
        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        zip_filename = filename.replace('.csv', '.zip')
        zip_path = os.path.join(self.output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, filename)
            zipf.write(metadata_path, metadata_filename)
        
        total_correct = sum(r.correct_predictions for r in results)
        total_examples = sum(r.total_examples for r in results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
        
        logger.info(f"CSV submission saved to: {csv_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Submission package saved to: {zip_path}")
        logger.info(f"Overall accuracy: {overall_accuracy:.2%} ({total_correct}/{total_examples})")
        
        return zip_path
    
    def save_submission_with_metadata(self, results: List[EvaluationResult], 
                                     metadata: Dict = None, filename: str = "submission.csv",
                                     config_path: str = None, args: argparse.Namespace = None):
        """
        Save submission with metadata
        """
        dataset_examples = getattr(self, '_last_dataset_examples', [])
        return self.save_submission(results, filename, metadata, dataset_examples, config_path, args)
    
    def list_datasets(self):
        """List available datasets"""
        print("Available Datasets:")
        print("-" * 50)
        for name, config in self.datasets.items():
            print(f"  {name}: {config['description']}")

    def get_metadata(self, config_path: str = None, args: argparse.Namespace = None, 
                    fallback_metadata: Dict = None) -> Dict:
        """
        Get metadata from various sources
        """
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": "MedGemma",
            "track": "internal_reasoning",
            "base_model_type": "OpenWeighted",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using MedGemma evaluation framework"
        }
        
        if fallback_metadata:
            metadata.update(fallback_metadata)
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config_metadata = config.get('metadata', config.get('meta_data', {}))
                metadata.update(config_metadata)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        
        if args:
            arg_fields = ['model_name', 'track', 'base_model_type', 
                         'base_model_name', 'dataset', 'additional_info']
            for field in arg_fields:
                if hasattr(args, field) and getattr(args, field) is not None:
                    metadata[field] = getattr(args, field)
        
        return metadata


def create_metadata_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for metadata
    """
    parser = argparse.ArgumentParser(description='MedGemma Evaluation Framework')
    
    parser.add_argument('--model-name', type=str, help='Name of the model')
    parser.add_argument('--base-model-name', type=str, help='Name of the base model')
    parser.add_argument('--base-model-type', type=str, choices=['API', 'OpenWeighted'], 
                       help='Type of base model')
    parser.add_argument('--track', type=str, choices=['internal_reasoning', 'agentic_reasoning'],
                       default='internal_reasoning', help='Competition track')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--additional-info', type=str, help='Additional information')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results', 
                       help='Output directory for results')
    parser.add_argument('--output-file', type=str, default='submission.csv', 
                       help='Output CSV filename')
    parser.add_argument('--subset-size', type=int, help='Limit evaluation to N examples')
    
    return parser


def load_config_file(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)


def load_and_merge_config(args):
    """Load config file and merge values into args"""
    if not args.config:
        return args
    
    config = load_config_file(args.config)
    
    if 'metadata' in config:
        metadata = config['metadata']
        for key, value in metadata.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    def add_config_to_args(config_dict, prefix=''):
        for key, value in config_dict.items():
            if key in ['metadata', 'dataset']:
                continue
            attr_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                add_config_to_args(value, attr_name)
            elif not hasattr(args, attr_name) or getattr(args, attr_name) is None:
                setattr(args, attr_name, value)
    
    add_config_to_args(config)
    return args
