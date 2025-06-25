from app import app
from app.models.scam_detector import ScamDetector
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
import nltk
import os
from collections import Counter
import numpy as np

def download_nltk_resources():
    flag_file = '.nltk_resources_downloaded'
    
    if not os.path.exists(flag_file):
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            print(f"Downloading {resource}")
            nltk.download(resource, quiet=True)
        
        with open(flag_file, 'w') as f:
            f.write('downloaded')

def create_ascii_bar_chart(data, title, width=60):
    """Create an ASCII bar chart"""
    if not data:
        return f"\n{title}\n" + "="*width + "\nNo data available\n"
    
    chart = f"\n{title}\n" + "="*width + "\n"
    
    max_value = max(data.values()) if data.values() else 1
    max_label_length = max(len(str(label)) for label in data.keys()) if data else 0
    
    for label, value in data.items():
        # Calculate bar length
        bar_length = int((value / max_value) * (width - max_label_length - 15)) if max_value > 0 else 0
        bar = '█' * bar_length
        
        # Format the line
        chart += f"{str(label):<{max_label_length}} │{bar:<{width-max_label_length-15}} {value:6.1f}\n"
    
    return chart

def create_ascii_line_graph(data, title, width=60, height=15):
    """Create an ASCII line graph"""
    if not data:
        return f"\n{title}\n" + "="*width + "\nNo data available\n"
    
    chart = f"\n{title}\n" + "="*width + "\n"
    
    values = list(data.values())
    labels = list(data.keys())
    
    if not values:
        return chart + "No data to display\n"
    
    max_value = max(values)
    min_value = min(values)
    value_range = max_value - min_value if max_value != min_value else 1
    
    # Create the graph grid
    graph_width = width - 15
    graph_height = height
    
    # Scale values to fit the graph height
    scaled_values = []
    for value in values:
        if value_range > 0:
            scaled = int(((value - min_value) / value_range) * (graph_height - 1))
        else:
            scaled = graph_height // 2
        scaled_values.append(scaled)
    
    # Create the graph
    for row in range(graph_height - 1, -1, -1):
        line = ""
        
        # Y-axis label
        y_value = min_value + (row / (graph_height - 1)) * value_range
        line += f"{y_value:6.1f} │"
        
        # Plot points
        for i, scaled_val in enumerate(scaled_values):
            if i < len(scaled_values) - 1:
                # Calculate spacing between points
                spacing = graph_width // (len(scaled_values) - 1) if len(scaled_values) > 1 else graph_width
                
                if scaled_val == row:
                    line += "●"
                elif i > 0 and scaled_values[i-1] != scaled_val:
                    # Draw connecting line
                    prev_val = scaled_values[i-1]
                    if (prev_val < row < scaled_val) or (scaled_val < row < prev_val):
                        line += "│"
                    else:
                        line += " "
                else:
                    line += " "
                
                # Add spacing
                line += " " * (spacing - 1)
            else:
                # Last point
                if scaled_val == row:
                    line += "●"
        
        chart += line + "\n"
    
    # X-axis
    chart += "       " + "─" * graph_width + "\n"
    
    # X-axis labels
    x_labels = "       "
    label_spacing = graph_width // len(labels) if labels else 1
    for i, label in enumerate(labels):
        if i < len(labels) - 1:
            x_labels += f"{label[:8]:<{label_spacing}}"
        else:
            x_labels += label[:8]
    chart += x_labels + "\n"
    
    return chart

def create_ascii_histogram(data, title, width=60):
    """Create an ASCII histogram"""
    if not data:
        return f"\n{title}\n" + "="*width + "\nNo data available\n"
    
    chart = f"\n{title}\n" + "="*width + "\n"
    
    max_value = max(data.values()) if data.values() else 1
    max_label_length = max(len(str(label)) for label in data.keys()) if data else 0
    
    # Calculate bar height (vertical bars)
    max_height = 10
    
    # Create vertical bars
    for height_level in range(max_height, 0, -1):
        line = " " * (max_label_length + 3)
        
        for label, value in data.items():
            bar_height = int((value / max_value) * max_height) if max_value > 0 else 0
            
            if bar_height >= height_level:
                line += "██ "
            else:
                line += "   "
        
        chart += line + "\n"
    
    # Base line
    chart += " " * (max_label_length + 3) + "─" * (len(data) * 3) + "\n"
    
    # Labels
    label_line = " " * (max_label_length + 3)
    for label in data.keys():
        label_line += f"{str(label)[:2]:<3}"
    chart += label_line + "\n"
    
    # Values
    value_line = " " * (max_label_length + 3)
    for value in data.values():
        value_line += f"{value:2.0f} "
    chart += value_line + "\n"
    
    return chart

def create_confusion_matrix_visual(tn, fp, fn, tp):
    """Create a visual confusion matrix"""
    chart = "\nConfusion Matrix Visualization:\n"
    chart += "="*40 + "\n"
    chart += "                 Predicted\n"
    chart += "               Legit  Scam\n"
    chart += "         ┌─────────────────┐\n"
    chart += f"  Legit  │  {tn:3d}  │  {fp:3d}  │\n"
    chart += "  Actual ├─────────────────┤\n"
    chart += f"  Scam   │  {fn:3d}  │  {tp:3d}  │\n"
    chart += "         └─────────────────┘\n"
    chart += "\nInterpretation:\n"
    chart += f"✓ True Negatives (TN): {tn} - Correctly identified legitimate messages\n"
    chart += f"✗ False Positives (FP): {fp} - Legitimate messages wrongly flagged as scams\n"
    chart += f"✗ False Negatives (FN): {fn} - Scams that were missed\n"
    chart += f"✓ True Positives (TP): {tp} - Correctly identified scams\n"
    
    return chart

def create_accuracy_gauge(accuracy):
    """Create an ASCII accuracy gauge"""
    gauge = "\nAccuracy Gauge:\n"
    gauge += "="*50 + "\n"
    
    # Create a horizontal gauge
    gauge_width = 40
    filled = int((accuracy / 100) * gauge_width)
    empty = gauge_width - filled
    
    gauge += "0%  "
    gauge += "█" * filled
    gauge += "░" * empty
    gauge += "  100%\n"
    gauge += f"     {accuracy:.1f}% Overall Accuracy\n"
    
    # Add performance indicator
    if accuracy >= 90:
        gauge += "     🟢 EXCELLENT Performance!\n"
    elif accuracy >= 80:
        gauge += "     🟡 GOOD Performance!\n"
    elif accuracy >= 70:
        gauge += "     🟠 FAIR Performance\n"
    else:
        gauge += "     🔴 NEEDS IMPROVEMENT\n"
    
    return gauge

def run_startup_accuracy_test():
    """Run a comprehensive accuracy test using k-fold cross validation and holdout test set"""
    print("\n" + "="*80)
    print("SCAM DETECTION SYSTEM - COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    try:
        # Initialize the scam detector
        detector = ScamDetector()
        total_samples = len(detector.train_data)
        
        print(f"\n📊 Dataset Overview:")
        print(f"Total samples: {total_samples}")
        scam_samples = len(detector.train_data[detector.train_data['is_scam'] == True])
        legit_samples = len(detector.train_data[detector.train_data['is_scam'] == False])
        print(f"Scam samples: {scam_samples} ({scam_samples/total_samples*100:.1f}%)")
        print(f"Legitimate samples: {legit_samples} ({legit_samples/total_samples*100:.1f}%)")
        
        if total_samples < 20:
            print("\n⚠️ WARNING: Dataset too small for reliable evaluation")
            print("Minimum recommended size: 20 samples")
            print("Current size:", total_samples)
            print("\nRecommendations:")
            print("1. Add more training data through the web interface")
            print("2. Include diverse examples of both scam and legitimate messages")
            return None
            
        # Prepare data for evaluation
        X = detector.vectorizer.fit_transform([
            detector.preprocess_text(text)['processed_text'] 
            for text in detector.train_data['text']
        ])
        y = detector.train_data['is_scam']
        
        print("\n🔄 Performing Cross-Validation...")
        
        # Perform stratified k-fold cross-validation
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            detector.model.fit(X_train, y_train)
            
            # Predict
            y_pred = detector.model.predict(X_val)
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            cv_scores['precision'].append(precision_score(y_val, y_pred))
            cv_scores['recall'].append(recall_score(y_val, y_pred))
            cv_scores['f1'].append(f1_score(y_val, y_pred))
            
            print(f"\nFold {fold} Results:")
            print(f"Accuracy:  {cv_scores['accuracy'][-1]:.3f}")
            print(f"Precision: {cv_scores['precision'][-1]:.3f}")
            print(f"Recall:    {cv_scores['recall'][-1]:.3f}")
            print(f"F1 Score:  {cv_scores['f1'][-1]:.3f}")
        
        # Calculate and display average metrics
        print("\n📈 Cross-Validation Summary:")
        print("-" * 50)
        for metric in cv_scores:
            mean = np.mean(cv_scores[metric])
            std = np.std(cv_scores[metric])
            print(f"{metric.capitalize():9} = {mean:.3f} ± {std:.3f}")
        
        # Perform final evaluation on holdout test set
        print("\n🎯 Final Evaluation on Holdout Test Set:")
        print("-" * 50)
        
        # Split into final train/test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train on final training set
        detector.model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = detector.model.predict(X_test)
        
        # Calculate final metrics
        final_accuracy = accuracy_score(y_test, y_pred)
        final_precision = precision_score(y_test, y_pred)
        final_recall = recall_score(y_test, y_pred)
        final_f1 = f1_score(y_test, y_pred)
        
        print(f"Test Set Size: {len(y_test)} samples")
        print(f"Final Accuracy:  {final_accuracy:.3f}")
        print(f"Final Precision: {final_precision:.3f}")
        print(f"Final Recall:    {final_recall:.3f}")
        print(f"Final F1 Score:  {final_f1:.3f}")
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        print("\n📊 Confusion Matrix Analysis:")
        print("-" * 50)
        print(create_confusion_matrix_visual(tn, fp, fn, tp))
        
        # Model reliability assessment
        print("\n🎯 Model Reliability Assessment:")
        print("-" * 50)
        
        reliability_score = 0
        reliability_factors = []
        
        # Factor 1: Dataset size
        if total_samples >= 100:
            reliability_score += 30
            reliability_factors.append("✓ Dataset size is adequate (>= 100 samples)")
        elif total_samples >= 50:
            reliability_score += 20
            reliability_factors.append("⚠️ Dataset size is moderate (>= 50 samples)")
        else:
            reliability_score += 10
            reliability_factors.append("❌ Dataset size is small (< 50 samples)")
        
        # Factor 2: Class balance
        class_ratio = min(scam_samples, legit_samples) / max(scam_samples, legit_samples)
        if class_ratio >= 0.8:
            reliability_score += 25
            reliability_factors.append("✓ Classes are well-balanced (ratio >= 0.8)")
        elif class_ratio >= 0.5:
            reliability_score += 15
            reliability_factors.append("⚠️ Classes are moderately balanced (ratio >= 0.5)")
        else:
            reliability_score += 5
            reliability_factors.append("❌ Classes are imbalanced (ratio < 0.5)")
        
        # Factor 3: Cross-validation stability
        cv_std = np.std(cv_scores['accuracy'])
        if cv_std < 0.05:
            reliability_score += 25
            reliability_factors.append("✓ Cross-validation results are stable (std < 0.05)")
        elif cv_std < 0.1:
            reliability_score += 15
            reliability_factors.append("⚠️ Cross-validation results show moderate variance (std < 0.1)")
        else:
            reliability_score += 5
            reliability_factors.append("❌ Cross-validation results show high variance (std >= 0.1)")
        
        # Factor 4: Performance consistency
        if abs(final_accuracy - np.mean(cv_scores['accuracy'])) < 0.05:
            reliability_score += 20
            reliability_factors.append("✓ Test set performance is consistent with cross-validation")
        else:
            reliability_score += 10
            reliability_factors.append("⚠️ Test set performance differs from cross-validation")
        
        print(f"\nReliability Score: {reliability_score}/100")
        print("\nReliability Factors:")
        for factor in reliability_factors:
            print(f"  {factor}")
        
        print("\n📝 Recommendations:")
        if reliability_score < 60:
            print("1. Add more training data (aim for at least 100 samples)")
            print("2. Balance the dataset with more samples from the minority class")
            print("3. Include more diverse examples of both scam and legitimate messages")
        elif reliability_score < 80:
            print("1. Consider adding more samples to improve model stability")
            print("2. Monitor performance on new data to ensure consistency")
        else:
            print("1. Continue monitoring model performance")
            print("2. Regularly update training data with new examples")
        
        print("\n" + "="*80)
        
        return {
            'accuracy': final_accuracy,
            'precision': final_precision,
            'recall': final_recall,
            'f1': final_f1,
            'confusion_matrix': conf_matrix.tolist(),
            'test_size': len(y_test),
            'reliability_score': reliability_score
        }

    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        return None

def generate_ml_report():
    """Generate a comprehensive ML system analysis report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE ML SYSTEM ANALYSIS REPORT")
    print("="*80)
    
    try:
        # Initialize the detector
        detector = ScamDetector()
        
        # 1. System Configuration Analysis
        print("\n1. SYSTEM CONFIGURATION")
        print("-"*50)
        print("TF-IDF Configuration:")
        print(f"- Max Features: {detector.vectorizer.max_features}")
        print(f"- Min Document Frequency: {detector.vectorizer.min_df}")
        print(f"- Max Document Frequency: {detector.vectorizer.max_df}")
        
        print("\nN-gram Configuration:")
        print("Bigram Vectorizer:")
        print(f"- N-gram Range: (2,2)")
        print(f"- Max Features: {detector.bigram_vectorizer.max_features}")
        print("\nTrigram Vectorizer:")
        print(f"- N-gram Range: (3,3)")
        print(f"- Max Features: {detector.trigram_vectorizer.max_features}")
        
        # 2. N-gram Analysis
        print("\n2. N-GRAM ANALYSIS")
        print("-"*50)
        print("Known Suspicious Patterns:")
        print(f"Total Bigrams: {len(detector.suspicious_bigrams)}")
        print(f"Total Trigrams: {len(detector.suspicious_trigrams)}")
        
        print("\nTop 10 Highest Scoring Bigrams:")
        sorted_bigrams = sorted(detector.suspicious_bigrams.items(), key=lambda x: x[1], reverse=True)[:10]
        for bigram, score in sorted_bigrams:
            print(f"- '{bigram}' (Score: {score:.2f})")
            
        print("\nTop 10 Highest Scoring Trigrams:")
        sorted_trigrams = sorted(detector.suspicious_trigrams.items(), key=lambda x: x[1], reverse=True)[:10]
        for trigram, score in sorted_trigrams:
            print(f"- '{trigram}' (Score: {score:.2f})")
        
        # 3. Training Data Analysis
        print("\n3. TRAINING DATA ANALYSIS")
        print("-"*50)
        train_data = detector.train_data
        total_samples = len(train_data)
        scam_samples = len(train_data[train_data['is_scam'] == True])
        legitimate_samples = len(train_data[train_data['is_scam'] == False])
        
        print(f"Total Training Samples: {total_samples}")
        print(f"Scam Samples: {scam_samples} ({(scam_samples/total_samples)*100:.1f}%)")
        print(f"Legitimate Samples: {legitimate_samples} ({(legitimate_samples/total_samples)*100:.1f}%)")
        
        # 4. Model Performance Analysis
        print("\n4. MODEL PERFORMANCE ANALYSIS")
        print("-"*50)
        
        # Split data for testing
        if len(train_data) >= 20:
            train_split, test_split = train_test_split(
                train_data, 
                test_size=0.2, 
                random_state=42,
                stratify=train_data['is_scam']
            )
            
            # Retrain on split
            detector.train_data = train_split
            detector.train_model()
            
            # Test predictions
            predictions = []
            actual_labels = []
            confidences = []
            scam_types = []
            
            print("\nRunning predictions on test set...")
            for _, row in test_split.iterrows():
                result = detector.predict(row['text'])
                predictions.append(result['confidence'] >= 40)  # Using 40% threshold as per system
                actual_labels.append(row['is_scam'])
                confidences.append(result['confidence'])
                scam_types.append(result.get('scam_type', 'Unknown'))
            
            # Calculate metrics
            accuracy = accuracy_score(actual_labels, predictions)
            precision = precision_score(actual_labels, predictions, zero_division=0)
            recall = recall_score(actual_labels, predictions, zero_division=0)
            f1 = f1_score(actual_labels, predictions, zero_division=0)
            conf_matrix = confusion_matrix(actual_labels, predictions)
            
            print("\nCore Metrics:")
            print(f"Accuracy: {accuracy*100:.2f}%")
            print(f"Precision: {precision*100:.2f}%")
            print(f"Recall: {recall*100:.2f}%")
            print(f"F1 Score: {f1*100:.2f}%")
            
            print("\nConfusion Matrix:")
            print("                  Predicted NO SCAM  Predicted SCAM")
            print(f"Actual NO SCAM    {conf_matrix[0][0]:^14d}    {conf_matrix[0][1]:^14d}")
            print(f"Actual SCAM       {conf_matrix[1][0]:^14d}    {conf_matrix[1][1]:^14d}")
            
            print("\nConfidence Distribution:")
            conf_ranges = [(0,20), (20,40), (40,60), (60,80), (80,100)]
            for low, high in conf_ranges:
                count = sum(1 for c in confidences if low <= c < high)
                print(f"{low:2d}-{high:2d}%: {'#'*(count*50//len(confidences))} ({count} predictions)")
            
            print("\nScam Type Distribution:")
            type_counts = Counter(scam_types)
            for stype, count in type_counts.most_common():
                print(f"{stype}: {count} instances ({count*100/len(scam_types):.1f}%)")
        
        # 5. Save Detailed Report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"ml_analysis_report_{timestamp}.txt"
        
        with open(report_file, "w") as f:
            f.write("MACHINE LEARNING SYSTEM ANALYSIS REPORT\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. SYSTEM CONFIGURATION\n")
            f.write("-"*50 + "\n")
            f.write(f"TF-IDF Max Features: {detector.vectorizer.max_features}\n")
            f.write(f"Bigram Max Features: {detector.bigram_vectorizer.max_features}\n")
            f.write(f"Trigram Max Features: {detector.trigram_vectorizer.max_features}\n\n")
            
            f.write("2. N-GRAM PATTERNS\n")
            f.write("-"*50 + "\n")
            f.write(f"Total Bigrams: {len(detector.suspicious_bigrams)}\n")
            f.write(f"Total Trigrams: {len(detector.suspicious_trigrams)}\n\n")
            
            f.write("Top 20 Bigrams:\n")
            for bigram, score in sorted_bigrams[:20]:
                f.write(f"- '{bigram}': {score:.2f}\n")
            f.write("\n")
            
            f.write("Top 20 Trigrams:\n")
            for trigram, score in sorted_trigrams[:20]:
                f.write(f"- '{trigram}': {score:.2f}\n")
            f.write("\n")
            
            f.write("3. MODEL PERFORMANCE\n")
            f.write("-"*50 + "\n")
            f.write(f"Accuracy: {accuracy*100:.2f}%\n")
            f.write(f"Precision: {precision*100:.2f}%\n")
            f.write(f"Recall: {recall*100:.2f}%\n")
            f.write(f"F1 Score: {f1*100:.2f}%\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(str(conf_matrix) + "\n\n")
            
            f.write("4. DATASET STATISTICS\n")
            f.write("-"*50 + "\n")
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"Scam Samples: {scam_samples}\n")
            f.write(f"Legitimate Samples: {legitimate_samples}\n")
        
        print(f"\n✓ Detailed analysis report saved to: {report_file}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Download required NLTK resources only once
    download_nltk_resources()
    
    # Run accuracy test on YOUR data
    run_startup_accuracy_test()
    
    # Start Flask app
    print(f"\n🚀 Starting Flask application...")
    print(f"📊 Access the web interface at: http://127.0.0.1:5000")
    print(f"🔍 Scam Detection System is ready!")
    
    # Run without debugger PIN and without reloader
    app.run(
        debug=True, 
        use_debugger=True,
        use_reloader=False,
        passthrough_errors=False
    )
