#!/usr/bin/env python3
"""
Improved LM-KBC Evaluation Script

This script provides comprehensive evaluation metrics, error analysis, and improvement suggestions
for LM-KBC 2025 predictions beyond basic metrics.

Usage:
    python improved_evaluate.py -p predictions.jsonl -g ground_truth.jsonl -o results/
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Relation type definitions
RELATION_TYPE = {
    "awardWonBy": "string",
    "hasCapacity": "numeric", 
    "hasArea": "numeric",
    "countryLandBordersCountry": "string",
    "personHasCityOfDeath": "string",
    "companyTradesAtStockExchange": "string"
}


def read_jsonl_file(file_path: Union[str, Path]) -> List[Dict]:
    """Read data from JSONL file."""
    with open(file_path, "r", encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    return rows


def analyze_prediction_patterns(pred_rows: List[Dict], gt_rows: List[Dict], max_examples: int = 10) -> Dict:
    """Analyze prediction patterns and error types."""
    analysis = {
        "relation_stats": {},
        "error_examples": {},
        "prediction_lengths": {},
        "empty_prediction_analysis": {}
    }
    
    # Group by relation
    pred_dict = {(r["SubjectEntity"], r["Relation"]): r["ObjectEntities"] for r in pred_rows}
    gt_dict = {(r["SubjectEntity"], r["Relation"]): r["ObjectEntities"] for r in gt_rows}
    
    for relation in RELATION_TYPE.keys():
        relation_preds = [(k, v) for k, v in pred_dict.items() if k[1] == relation]
        relation_gts = [(k, v) for k, v in gt_dict.items() if k[1] == relation]
        
        # Prediction length analysis
        pred_lengths = [len(pred[1]) for pred in relation_preds]
        gt_lengths = [len(gt[1]) for gt in relation_gts if gt[0] in [p[0] for p in relation_preds]]
        
        analysis["prediction_lengths"][relation] = {
            "avg_pred_length": sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0,
            "avg_gt_length": sum(gt_lengths) / len(gt_lengths) if gt_lengths else 0,
            "max_pred": max(pred_lengths) if pred_lengths else 0,
            "max_gt": max(gt_lengths) if gt_lengths else 0
        }
        
        # Empty prediction analysis
        empty_preds = [pred for pred in relation_preds if len(pred[1]) == 0]
        empty_gts = [gt for gt in relation_gts if len(gt[1]) == 0 and gt[0] in [p[0] for p in relation_preds]]
        
        analysis["empty_prediction_analysis"][relation] = {
            "empty_predictions": len(empty_preds),
            "should_be_empty": len(empty_gts),
            "total_predictions": len(relation_preds)
        }
        
        # Error example collection
        analysis["error_examples"][relation] = []
        error_count = 0
        
        for pred_key, pred_entities in relation_preds:
            if error_count >= max_examples:
                break
                
            gt_entities = gt_dict.get(pred_key, [])
            
            if pred_entities != gt_entities:  # Only collect error examples
                analysis["error_examples"][relation].append({
                    "entity": pred_key[0],
                    "predicted": pred_entities,
                    "ground_truth": gt_entities,
                    "missing": list(set(gt_entities) - set(pred_entities)),
                    "extra": list(set(pred_entities) - set(gt_entities)),
                    "error_type": classify_error_type(pred_entities, gt_entities)
                })
                error_count += 1
    
    return analysis


def classify_error_type(pred_entities: List[str], gt_entities: List[str]) -> str:
    """Classify error types."""
    if len(pred_entities) == 0 and len(gt_entities) > 0:
        return "Complete Miss"
    elif len(pred_entities) > 0 and len(gt_entities) == 0:
        return "False Positive"
    elif len(pred_entities) < len(gt_entities):
        return "Partial Miss"
    elif len(pred_entities) > len(gt_entities):
        return "Over Prediction"
    else:
        return "Content Error"


def calculate_detailed_metrics(pred_rows: List[Dict], gt_rows: List[Dict]) -> Dict:
    """Calculate detailed evaluation metrics."""
    try:
        from evaluate import evaluate_per_sr_pair, macro_average_per_relation, micro_average_per_relation, prediction_statistics
        
        # Basic metrics
        scores_per_sr = evaluate_per_sr_pair(pred_rows, gt_rows, RELATION_TYPE)
        macro_metrics = macro_average_per_relation(scores_per_sr)
        micro_metrics = micro_average_per_relation(scores_per_sr)
        stats = prediction_statistics(scores_per_sr)
        
        return {
            "per_sample": scores_per_sr,
            "macro": macro_metrics,
            "micro": micro_metrics,
            "stats": stats
        }
    except ImportError:
        print("Warning: Official evaluation module not found. Using simplified metrics.")
        return calculate_simplified_metrics(pred_rows, gt_rows)


def calculate_simplified_metrics(pred_rows: List[Dict], gt_rows: List[Dict]) -> Dict:
    """Calculate simplified metrics when official evaluation module is not available."""
    # Create mapping
    pred_dict = {(r["SubjectEntity"], r["Relation"]): r["ObjectEntities"] for r in pred_rows}
    gt_dict = {(r["SubjectEntity"], r["Relation"]): r["ObjectEntities"] for r in gt_rows}
    
    relation_metrics = {}
    
    for relation in RELATION_TYPE.keys():
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for key in pred_dict:
            if key[1] == relation and key in gt_dict:
                pred_set = set(pred_dict[key])
                gt_set = set(gt_dict[key])
                
                if len(pred_set) == 0 and len(gt_set) == 0:
                    precision, recall, f1 = 1.0, 1.0, 1.0
                elif len(pred_set) == 0:
                    precision, recall, f1 = 0.0, 0.0, 0.0
                elif len(gt_set) == 0:
                    precision, recall, f1 = 0.0, 0.0, 0.0
                else:
                    intersection = pred_set & gt_set
                    precision = len(intersection) / len(pred_set)
                    recall = len(intersection) / len(gt_set)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
        
        relation_metrics[relation] = {
            "macro-p": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
            "macro-r": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
            "macro-f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        }
    
    # Overall metrics
    all_precisions = [v["macro-p"] for v in relation_metrics.values()]
    all_recalls = [v["macro-r"] for v in relation_metrics.values()]
    all_f1s = [v["macro-f1"] for v in relation_metrics.values()]
    
    relation_metrics["*** All Relations ***"] = {
        "macro-p": sum(all_precisions) / len(all_precisions),
        "macro-r": sum(all_recalls) / len(all_recalls),
        "macro-f1": sum(all_f1s) / len(all_f1s)
    }
    
    return {
        "macro": relation_metrics,
        "micro": relation_metrics,  # Simplified
        "stats": {rel: {"avg. #preds": 0, "#empty preds": 0} for rel in RELATION_TYPE.keys()}
    }


def generate_improvement_suggestions(metrics: Dict, analysis: Dict) -> List[str]:
    """Generate specific improvement suggestions based on results."""
    suggestions = []
    
    for relation, macro_scores in metrics["macro"].items():
        if relation == "*** All Relations ***":
            continue
            
        precision = macro_scores["macro-p"]
        recall = macro_scores["macro-r"]
        f1 = macro_scores["macro-f1"]
        
        # Get prediction length information
        pred_length_info = analysis["prediction_lengths"].get(relation, {})
        avg_pred = pred_length_info.get("avg_pred_length", 0)
        avg_gt = pred_length_info.get("avg_gt_length", 0)
        
        if f1 < 0.3:
            suggestions.append(f" {relation}: Critical Issue (F1={f1:.3f})")
            if recall < 0.1:
                suggestions.append(f"   - Extremely low recall, consider more aggressive strategies or divide-and-conquer methods")
            if avg_pred < avg_gt * 0.5:
                suggestions.append(f"   - Insufficient predictions (pred: {avg_pred:.1f} vs actual: {avg_gt:.1f}), increase output length")
        
        elif f1 < 0.6:
            suggestions.append(f" {relation}: Needs Improvement (F1={f1:.3f})")
            if precision < 0.5:
                suggestions.append(f"   - Low precision, improve post-processing to filter incorrect answers")
            if recall < 0.5:
                suggestions.append(f"   - Low recall, try multi-round queries or different prompt strategies")
        
        else:
            suggestions.append(f" {relation}: Good Performance (F1={f1:.3f})")
    
    # Overall suggestions
    overall_f1 = metrics["macro"]["*** All Relations ***"]["macro-f1"]
    if overall_f1 < 0.5:
        suggestions.append("\n Overall Improvement Strategy:")
        suggestions.append("   - Consider relation-specific strategy selection")
        suggestions.append("   - Implement hybrid methods combining multiple strategies")
        suggestions.append("   - Focus optimization on poorly performing relations")
    
    return suggestions


def create_performance_plots(metrics: Dict, output_dir: Path):
    """Create performance visualization charts."""
    # Set font for plots
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # Extract relation-level metrics
    relations = [rel for rel in metrics["macro"].keys() if rel != "*** All Relations ***"]
    precisions = [metrics["macro"][rel]["macro-p"] for rel in relations]
    recalls = [metrics["macro"][rel]["macro-r"] for rel in relations]
    f1s = [metrics["macro"][rel]["macro-f1"] for rel in relations]
    
    # Create performance comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Precision, Recall, F1 comparison
    x = range(len(relations))
    axes[0, 0].bar([i-0.2 for i in x], precisions, 0.2, label='Precision', alpha=0.8)
    axes[0, 0].bar(x, recalls, 0.2, label='Recall', alpha=0.8)
    axes[0, 0].bar([i+0.2 for i in x], f1s, 0.2, label='F1', alpha=0.8)
    axes[0, 0].set_title('Performance by Relation')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([rel.replace('companyTradesAtStockExchange', 'companyTrades') for rel in relations], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    
    # F1 score heatmap
    f1_data = [[f1] for f1 in f1s]
    im = axes[0, 1].imshow(f1_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('F1 Scores Heatmap')
    axes[0, 1].set_yticks(range(len(relations)))
    axes[0, 1].set_yticklabels([rel.replace('companyTradesAtStockExchange', 'companyTrades') for rel in relations])
    axes[0, 1].set_xticks([])
    plt.colorbar(im, ax=axes[0, 1])
    
    # Performance scatter plot
    axes[1, 0].scatter(precisions, recalls, s=100, alpha=0.7)
    for i, rel in enumerate(relations):
        axes[1, 0].annotate(rel.replace('companyTradesAtStockExchange', 'companyTrades'), 
                           (precisions[i], recalls[i]), fontsize=8)
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Precision vs Recall')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    # F1 scores by relation
    axes[1, 1].bar(range(len(relations)), f1s, alpha=0.7)
    axes[1, 1].set_title('F1 Scores by Relation')
    axes[1, 1].set_xticks(range(len(relations)))
    axes[1, 1].set_xticklabels([rel.replace('companyTradesAtStockExchange', 'companyTrades') for rel in relations], rotation=45)
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_report(metrics: Dict, analysis: Dict, suggestions: List[str], output_dir: Path, 
                        pred_file: str, gt_file: str):
    """Save detailed analysis report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate markdown report
    report_content = f"""# LM-KBC Evaluation Report
    
## Basic Information
- **Evaluation Time**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Prediction File**: {pred_file}
- **Ground Truth File**: {gt_file}

## Overall Performance Summary
- **Macro Average Precision**: {metrics['macro']['*** All Relations ***']['macro-p']:.3f}
- **Macro Average Recall**: {metrics['macro']['*** All Relations ***']['macro-r']:.3f}
- **Macro Average F1**: {metrics['macro']['*** All Relations ***']['macro-f1']:.3f}

## Detailed Analysis by Relation

"""
    
    for relation in RELATION_TYPE.keys():
        if relation in metrics['macro']:
            macro_scores = metrics['macro'][relation]
            length_info = analysis['prediction_lengths'][relation]
            
            report_content += f"""### {relation}
- **Precision**: {macro_scores['macro-p']:.3f}
- **Recall**: {macro_scores['macro-r']:.3f}  
- **F1**: {macro_scores['macro-f1']:.3f}
- **Average Prediction Length**: {length_info['avg_pred_length']:.1f} (Ground Truth: {length_info['avg_gt_length']:.1f})

"""
            
            # Add error examples
            if relation in analysis['error_examples'] and analysis['error_examples'][relation]:
                report_content += "**Error Examples**:\n"
                for i, example in enumerate(analysis['error_examples'][relation][:5]):
                    report_content += f"- Entity: {example['entity']}\n"
                    report_content += f"  - Predicted: {example['predicted']}\n"
                    report_content += f"  - Ground Truth: {example['ground_truth']}\n"
                    if example['missing']:
                        report_content += f"  - Missing: {example['missing']}\n"
                    if example['extra']:
                        report_content += f"  - Extra: {example['extra']}\n"
                report_content += "\n"
    
    # Add improvement suggestions
    report_content += "## Improvement Suggestions\n\n"
    for suggestion in suggestions:
        report_content += f"{suggestion}\n"
    
    # Save report
    with open(output_dir / f"evaluation_report_{timestamp}.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    # Save detailed data as JSON
    detailed_data = {
        "timestamp": timestamp,
        "metrics": metrics,
        "analysis": analysis,
        "suggestions": suggestions
    }
    
    with open(output_dir / f"detailed_results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=2, ensure_ascii=False)
    
    # Save CSV format results table
    macro_df = pd.DataFrame(metrics['macro']).transpose().round(3)
    macro_df.to_csv(output_dir / f"macro_results_{timestamp}.csv")
    
    print(f"\n Detailed reports saved to:")
    print(f"   - Markdown report: {output_dir / f'evaluation_report_{timestamp}.md'}")
    print(f"   - Detailed data: {output_dir / f'detailed_results_{timestamp}.json'}")
    print(f"   - CSV results: {output_dir / f'macro_results_{timestamp}.csv'}")
    print(f"   - Performance charts: {output_dir / 'performance_analysis.png'}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Enhanced LM-KBC Evaluation Script - Provides detailed analysis and improvement suggestions")
    
    parser.add_argument("-p", "--predictions", type=str, required=True,
                       help="Path to predictions file")
    parser.add_argument("-g", "--ground_truth", type=str, required=True,
                       help="Path to ground truth file") 
    parser.add_argument("-o", "--output_dir", type=str, default="evaluation_results",
                       help="Output directory (default: evaluation_results)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(" Starting evaluation analysis...")
    
    # Read data
    pred_rows = read_jsonl_file(args.predictions)
    gt_rows = read_jsonl_file(args.ground_truth)
    
    print(f" Loaded {len(pred_rows)} predictions and {len(gt_rows)} ground truth labels")
    
    # Calculate metrics
    print(" Calculating evaluation metrics...")
    metrics = calculate_detailed_metrics(pred_rows, gt_rows)
    
    # Analyze prediction patterns
    print(" Analyzing prediction patterns...")
    analysis = analyze_prediction_patterns(pred_rows, gt_rows)
    
    # Generate improvement suggestions
    suggestions = generate_improvement_suggestions(metrics, analysis)
    
    # Display basic results
    macro_df = pd.DataFrame(metrics['macro']).transpose().round(3)
    
    print("\n Evaluation Results:")
    print(macro_df)
    
    # Display improvement suggestions
    print("\n Improvement Suggestions:")
    for suggestion in suggestions:
        print(suggestion)
    
    # Create plots
    if not args.no_plots:
        print("\n Generating performance charts...")
        try:
            create_performance_plots(metrics, output_dir)
        except Exception as e:
            print(f" Chart generation failed: {e}")
    
    # Save detailed report
    print("\n Saving detailed report...")
    save_detailed_report(metrics, analysis, suggestions, output_dir, 
                        args.predictions, args.ground_truth)
    
    print("\n Evaluation completed!")


if __name__ == "__main__":
    main()