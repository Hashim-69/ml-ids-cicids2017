"""
CICIDS2017 Intrusion Detection System - Memory Optimized & Fixed
Author: Your Name
Description: Multi-class ML-based Network Intrusion Detection System
Detects: BENIGN traffic + Multiple attack types (DDoS, PortScan, Brute Force, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CICIDS2017_MultiClass_IDS:
    """
    Advanced Intrusion Detection System with Multi-Class Attack Detection
    NO SMOTE - Uses class weights instead for memory efficiency
    """
    
    def __init__(self, filepath, classification_type='multiclass', sample_size=None):
        """
        Initialize the IDS
        
        Args:
            filepath: Path to cleaned CICIDS2017 CSV file
            classification_type: 'binary' or 'multiclass'
            sample_size: Number of rows to load (None = all data)
        """
        self.filepath = filepath
        self.classification_type = classification_type
        self.sample_size = sample_size
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.attack_types = []
        
    def load_data(self):
        """Load and explore the dataset with sampling"""
        print("=" * 70)
        print("LOADING CICIDS2017 CLEANED DATASET")
        print("=" * 70)
        
        print(f"\n📂 Loading from: {self.filepath}")
        
        # Load with sampling if specified
        if self.sample_size:
            print(f"⚡ Sampling {self.sample_size:,} rows for memory efficiency...")
            self.df = pd.read_csv(self.filepath, nrows=self.sample_size)
        else:
            print(f"⚠️  Loading ALL data (may cause memory issues)...")
            self.df = pd.read_csv(self.filepath)
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Total Samples: {len(self.df):,}")
        print(f"  Features: {len(self.df.columns)}")
        print(f"  Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Find label column
        possible_labels = ['Attack Type', 'Label', 'Class', 'Attack', 'Type']
        self.label_col = None
        
        for col in possible_labels:
            if col in self.df.columns:
                self.label_col = col
                print(f"\n✓ Found label column: '{self.label_col}'")
                break
        
        if self.label_col is None:
            self.label_col = self.df.columns[-1]
            print(f"\n⚠️  Using last column as label: '{self.label_col}'")
        
        # Display attack distribution
        print(f"\n🎯 Attack Type Distribution:")
        print("=" * 70)
        attack_counts = self.df[self.label_col].value_counts()
        self.attack_types = attack_counts.index.tolist()
        
        for attack, count in attack_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {str(attack):<25} {count:>10,} ({percentage:>5.2f}%)")
        
        print(f"\n  Total Attack Categories: {len(self.attack_types)}")
        
        return self
    
    def preprocess_data(self):
        """Preprocess data WITHOUT SMOTE"""
        print("\n" + "=" * 70)
        print("PREPROCESSING DATA")
        print("=" * 70)
        
        # Data quality check
        print(f"\n🔍 Data Quality Check:")
        print(f"  Missing values: {self.df.isnull().sum().sum()}")
        print(f"  Duplicate rows: {self.df.duplicated().sum()}")
        
        # Handle issues
        if self.df.isnull().sum().sum() > 0:
            self.df = self.df.fillna(0)
            print("  ✓ Filled missing values")
        
        if self.df.duplicated().sum() > 0:
            self.df = self.df.drop_duplicates()
            print("  ✓ Removed duplicates")
        
        # Separate features and labels
        X = self.df.drop(columns=[self.label_col])
        y = self.df[self.label_col]
        
        # Classification setup
        if self.classification_type == 'binary':
            print(f"\n🎯 Binary Classification Mode: BENIGN vs ATTACK")
            y = y.apply(lambda x: 0 if 'BENIGN' in str(x).upper() or 'NORMAL' in str(x).upper() else 1)
            self.class_names = ['BENIGN', 'ATTACK']
        else:
            print(f"\n🎯 Multi-Class Classification Mode: Detecting {len(self.attack_types)} Attack Types")
            y = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_
        
        # Select numeric features only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        print(f"\n✓ Features: {len(numeric_cols)} numeric columns selected")
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Split data
        print(f"\n📊 Splitting data (80% train, 20% test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training samples: {len(self.X_train):,}")
        print(f"  Testing samples: {len(self.X_test):,}")
        
        # Scale features
        print(f"\n🔧 Scaling features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("  ✓ Features scaled with StandardScaler")
        
        # Show class distribution (NO SMOTE)
        print(f"\n⚖️  Class Distribution (Using class weights instead of SMOTE):")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for cls, cnt in zip(unique, counts):
            if self.classification_type == 'multiclass':
                print(f"    {self.class_names[cls]:<25} {cnt:>10,}")
            else:
                print(f"    {self.class_names[cls]:<25} {cnt:>10,}")
        
        print(f"\n  💡 Models will use class_weight='balanced' to handle imbalance")
        
        return self
    
    def train_models(self):
        """Train ML models with balanced class weights"""
        print("\n" + "=" * 70)
        print("TRAINING MACHINE LEARNING MODELS")
        print("=" * 70)
        
        # Define models with class weights
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42, 
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42, 
                max_depth=15,
                min_samples_split=5,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\n🤖 Training {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"  ✓ Training completed")
            
            # Make predictions
            print(f"  📊 Evaluating on test set...")
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            if self.classification_type == 'binary':
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                self.results[name] = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='binary'),
                    'recall': recall_score(self.y_test, y_pred, average='binary'),
                    'f1': f1_score(self.y_test, y_pred, average='binary'),
                    'auc': roc_auc_score(self.y_test, y_pred_proba),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
            else:
                self.results[name] = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'predictions': y_pred,
                    'probabilities': None
                }
            
            print(f"  ✓ Accuracy: {self.results[name]['accuracy']:.4f}")
            print(f"  ✓ F1-Score: {self.results[name]['f1']:.4f}")
        
        return self
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION RESULTS")
        print("=" * 70)
        
        # Create results dataframe
        results_data = {
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1'] for m in self.results]
        }
        
        if self.classification_type == 'binary':
            results_data['AUC-ROC'] = [self.results[m]['auc'] for m in self.results]
        
        results_df = pd.DataFrame(results_data)
        
        print("\n📊 Performance Metrics:")
        print(results_df.to_string(index=False))
        
        # Find best model
        best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {results_df.loc[results_df['F1-Score'].idxmax(), 'F1-Score']:.4f}")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report - {best_model_name}:")
        print("=" * 70)
        print(classification_report(
            self.y_test, 
            self.results[best_model_name]['predictions'],
            target_names=self.class_names,
            zero_division=0
        ))
        
        # Per-class performance for multi-class
        if self.classification_type == 'multiclass':
            print(f"\n🎯 Per-Attack Detection Performance ({best_model_name}):")
            print("=" * 70)
            y_pred = self.results[best_model_name]['predictions']
            
            for i, attack in enumerate(self.class_names):
                mask = self.y_test == i
                if mask.sum() > 0:
                    acc = accuracy_score(self.y_test[mask], y_pred[mask])
                    print(f"  {attack:<25} Detection Rate: {acc:.2%}")
        
        return self
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        if self.classification_type == 'binary':
            self._visualize_binary()
        else:
            self._visualize_multiclass()
        
        plt.tight_layout()
        plt.savefig('ids_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualizations saved as 'ids_results.png'")
        plt.show()
        
        return self
    
    def _visualize_binary(self):
        """Visualizations for binary classification"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Model Comparison
        ax1 = plt.subplot(2, 3, 1)
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1'] for m in self.results]
        })
        
        metrics_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
            kind='bar', ax=ax1, rot=45
        )
        ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)
        
        # 2-4. Confusion Matrices
        for idx, (name, results) in enumerate(list(self.results.items())[:3], start=2):
            ax = plt.subplot(2, 3, idx)
            cm = confusion_matrix(self.y_test, results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       cbar_kws={'label': 'Count'})
            ax.set_title(f'{name}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
        
        # 5. ROC Curves
        ax5 = plt.subplot(2, 3, 5)
        for name, results in self.results.items():
            if results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results['probabilities'])
                auc = results['auc']
                ax5.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax5.set_xlabel('False Positive Rate')
        ax5.set_ylabel('True Positive Rate')
        ax5.set_title('ROC Curves', fontsize=12, fontweight='bold')
        ax5.legend(loc='lower right', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Feature Importance
        ax6 = plt.subplot(2, 3, 6)
        rf_model = self.models['Random Forest']
        feature_importance = rf_model.feature_importances_
        top_10_idx = np.argsort(feature_importance)[-10:]
        
        ax6.barh(range(10), feature_importance[top_10_idx], color='steelblue')
        ax6.set_yticks(range(10))
        ax6.set_yticklabels([f'Feature {i}' for i in top_10_idx])
        ax6.set_xlabel('Importance')
        ax6.set_title('Top 10 Features (Random Forest)', fontsize=10, fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)
    
    def _visualize_multiclass(self):
        """Visualizations for multi-class classification"""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Model Comparison
        ax1 = plt.subplot(2, 3, 1)
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1'] for m in self.results]
        })
        
        metrics_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
            kind='bar', ax=ax1, rot=45
        )
        ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Best Model Confusion Matrix
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        ax2 = plt.subplot(2, 3, (2, 5))
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['predictions'])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2,
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Detection Rate'})
        ax2.set_title(f'Confusion Matrix - {best_model_name}\n(Normalized)', 
                     fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual Attack Type')
        ax2.set_xlabel('Predicted Attack Type')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
        
        # 3. Attack Distribution
        ax3 = plt.subplot(2, 3, 3)
        test_distribution = pd.Series(self.y_test).value_counts().sort_index()
        attack_names = [self.class_names[i] for i in test_distribution.index]
        ax3.barh(attack_names, test_distribution.values, color='coral')
        ax3.set_xlabel('Number of Samples')
        ax3.set_title('Test Set Distribution', fontsize=10, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Per-Class Accuracy
        ax4 = plt.subplot(2, 3, 6)
        y_pred = self.results[best_model_name]['predictions']
        per_class_acc = []
        
        for i in range(len(self.class_names)):
            mask = self.y_test == i
            if mask.sum() > 0:
                acc = accuracy_score(self.y_test[mask], y_pred[mask])
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0)
        
        colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in per_class_acc]
        ax4.barh(self.class_names, per_class_acc, color=colors)
        ax4.set_xlabel('Detection Rate')
        ax4.set_title(f'Per-Attack Detection Rate\n({best_model_name})', 
                     fontsize=10, fontweight='bold')
        ax4.set_xlim([0, 1])
        ax4.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (attack, acc) in enumerate(zip(self.class_names, per_class_acc)):
            ax4.text(acc + 0.02, i, f'{acc:.1%}', va='center', fontsize=8)
    
    def save_best_model(self, filename='best_ids_model.pkl'):
        """Save the best performing model"""
        import pickle
        
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        best_model = self.models[best_model_name]
        
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'model_name': best_model_name,
            'classification_type': self.classification_type,
            'class_names': self.class_names
        }
        
        if self.classification_type == 'multiclass':
            model_data['label_encoder'] = self.label_encoder
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Best model ({best_model_name}) saved as '{filename}'")
        print(f"   Classification Type: {self.classification_type}")
        print(f"   F1-Score: {self.results[best_model_name]['f1']:.4f}")
        
        return self


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 70)
    print("   CICIDS2017 ADVANCED INTRUSION DETECTION SYSTEM")
    print("=" * 70)
    
    # Dataset path
    filepath = r'C:\Users\Al Salik Computerz\Downloads\archive\cicids2017_cleaned.csv'
    
    # Classification type
    classification_type = 'multiclass'
    
    # CRITICAL: Set sample size to avoid memory errors
    # 100000 = 100k rows (Safe for 4GB RAM)
    # 250000 = 250k rows (Safe for 8GB RAM)
    # None = ALL data (Requires 32GB+ RAM)
    sample_size = 100000
    
    print(f"\n📂 Dataset: {filepath}")
    print(f"🎯 Mode: {classification_type.upper()} Classification")
    print(f"⚡ Sample Size: {sample_size:,} rows")
    
    # Initialize and run IDS with sample_size parameter
    ids = CICIDS2017_MultiClass_IDS(filepath, classification_type, sample_size)
    
    try:
        ids.load_data() \
           .preprocess_data() \
           .train_models() \
           .evaluate_models() \
           .visualize_results() \
           .save_best_model()
        
        print("\n" + "=" * 70)
        print("✅ INTRUSION DETECTION SYSTEM COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📁 Output Files Generated:")
        print("  • ids_results.png - Performance visualizations")
        print("  • best_ids_model.pkl - Trained model for deployment")
        print("\n💡 Resume Highlights:")
        print(f"  • Built ML-based IDS with {classification_type} classification")
        print(f"  • Trained on CICIDS2017 dataset with {len(ids.attack_types)} attack types")
        print(f"  • Achieved {max(ids.results.values(), key=lambda x: x['accuracy'])['accuracy']:.2%} accuracy")
        print("  • Used balanced class weights for imbalanced data")
        print("  • Compared multiple ML algorithms (RF, DT, GB)")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find file at:")
        print(f"   {filepath}")
        print("\n💡 Tips:")
        print("  1. Check if the file path is correct")
        print("  2. Make sure the dataset exists at that location")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()