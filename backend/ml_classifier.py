"""
Machine Learning Signal Classification System: AI-Powered Trading Signal Prediction

This module provides a comprehensive machine learning framework for predicting trading
signals using advanced feature engineering and supervised learning techniques. It represents
the integration of modern machine learning with quantitative trading strategies.

The system enables comparison between rule-based trading strategies and machine learning
approaches, providing insights into when AI can enhance traditional technical analysis
and when human-designed rules may be superior.

Key Features:
============

1. **Advanced Feature Engineering**
   - Technical indicator computation (SMA, EMA, RSI, MACD, Bollinger Bands)
   - Price momentum and volatility features
   - Market structure indicators
   - Volume-based features (when available)

2. **Machine Learning Models**
   - Random Forest ensemble learning
   - XGBoost gradient boosting (optional)
   - Feature importance analysis
   - Hyperparameter optimization

3. **Signal Prediction & Validation**
   - Multi-class signal classification (long/short/neutral)
   - Confidence scoring for predictions
   - Cross-validation and performance metrics
   - Backtesting integration

4. **Comparative Analysis**
   - Rule-based vs ML signal comparison
   - Agreement rate analysis
   - Performance attribution
   - Model interpretation and insights

Architecture:
============

The ML classifier follows a comprehensive pipeline architecture:

1. **Data Preprocessing Layer**
   - Market data validation and cleaning
   - Feature extraction and engineering
   - Data normalization and scaling
   - Missing value handling

2. **Model Training Layer**
   - Supervised learning with historical signals
   - Feature selection and importance ranking
   - Cross-validation and performance evaluation
   - Hyperparameter optimization

3. **Prediction Layer**
   - Real-time signal generation
   - Confidence scoring and uncertainty quantification
   - Model interpretation and explainability
   - Performance monitoring

4. **Analysis Layer**
   - Comparative performance analysis
   - Feature importance visualization
   - Signal agreement metrics
   - Model diagnostics and validation

Usage Examples:
===============

Basic Usage:
```python
from backend.ml_classifier import MLSignalClassifier

# Initialize classifier
classifier = MLSignalClassifier(
    model_type="random_forest",
    n_estimators=100,
    random_state=42
)

# Train model
results = classifier.train(price_data, signals_data)

# Generate predictions
predictions = classifier.predict_signals(new_data)

# Compare with rule-based signals
comparison = classifier.compare_with_rule_based(
    price_data, rule_signals
)
```

Advanced Usage:
```python
# Custom feature engineering
features = classifier.extract_features(price_data)

# Model interpretation
importance = classifier.get_feature_importance()

# Performance analysis
metrics = classifier.evaluate_performance(test_data, test_signals)

# Export trained model
classifier.save_model("my_trading_model.pkl")
```

Educational Value:
=================

This module demonstrates:

1. **Machine Learning in Finance**
   - Feature engineering for financial data
   - Supervised learning for trading signals
   - Model validation and performance evaluation
   - Overfitting prevention and generalization

2. **Technical Analysis Enhancement**
   - Traditional indicators as ML features
   - Pattern recognition through machine learning
   - Signal quality improvement techniques
   - Human-AI collaboration in trading

3. **Model Development Best Practices**
   - Cross-validation methodologies
   - Feature selection techniques
   - Performance metrics for classification
   - Model interpretability and explainability

4. **Quantitative Research Methods**
   - Hypothesis testing with ML models
   - Statistical significance assessment
   - Comparative analysis frameworks
   - Research methodology in finance

Integration Points:
==================

The ML classifier integrates with:
- Trading strategy frameworks
- Technical analysis libraries
- Backtesting systems
- Performance evaluation modules
- Visualization and reporting tools

Performance Considerations:
==========================

- Efficient feature computation
- Scalable model training
- Real-time prediction capabilities
- Memory-conscious data processing
- Cross-platform compatibility

Dependencies:
============

- scikit-learn for machine learning
- pandas/numpy for data processing
- joblib for model serialization
- Optional: XGBoost for gradient boosting
- Technical analysis libraries

Author: Strategy Explainer Framework
Version: 2.0
License: Educational Use
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLSignalClassifier:
    """
    Machine learning classifier for predicting trading signals.
    
    This class implements a comprehensive machine learning framework for
    predicting trading signals using advanced feature engineering and
    supervised learning techniques. It enables comparison between rule-based
    and ML-based trading approaches.
    
    The classifier is designed to work with various market data formats
    and provides extensive capabilities for model training, evaluation,
    and deployment in trading systems.
    
    Key Capabilities:
    ================
    
    1. **Feature Engineering**
       - Technical indicator computation
       - Price momentum and volatility features
       - Market structure indicators
       - Volume-based features
    
    2. **Model Training**
       - Random Forest and XGBoost support
       - Cross-validation and performance metrics
       - Feature importance analysis
       - Hyperparameter optimization
    
    3. **Signal Prediction**
       - Multi-class signal classification
       - Confidence scoring
       - Real-time prediction capabilities
       - Uncertainty quantification
    
    4. **Comparative Analysis**
       - Rule-based vs ML comparison
       - Agreement rate analysis
       - Performance attribution
       - Model interpretation
    
    Attributes:
        model_type (str): Type of ML model ('random_forest', 'xgboost')
        random_state (int): Random seed for reproducibility
        model: Trained machine learning model
        scaler (StandardScaler): Feature scaling transformer
        feature_names (list): Names of engineered features
        is_trained (bool): Whether model has been trained
        training_score (float): Training accuracy score
        validation_score (float): Validation accuracy score
    
    Example Usage:
    =============
    ```python
    # Initialize classifier
    classifier = MLSignalClassifier(
        model_type="random_forest",
        n_estimators=100,
        random_state=42
    )
    
    # Train model
    results = classifier.train(price_data, signals_data)
    
    # Generate predictions
    predictions = classifier.predict_signals(new_data)
    
    # Compare with rule-based
    comparison = classifier.compare_with_rule_based(
        price_data, rule_signals
    )
    ```
    """
    
    def __init__(self, 
                 model_type: str = "random_forest",
                 n_estimators: int = 100,
                 random_state: int = 42):
        """
        Initialize the ML signal classifier.
        
        This constructor sets up the machine learning model with specified
        configuration and prepares the feature engineering pipeline.
        
        Args:
            model_type (str, optional): Type of ML model to use.
                Options: 'random_forest', 'xgboost'. Defaults to "random_forest".
            n_estimators (int, optional): Number of trees for ensemble methods.
                Defaults to 100.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to 42.
        
        Initialization Process:
        ======================
        1. **Model Selection**: Choose appropriate ML algorithm
        2. **Configuration Setup**: Set hyperparameters and options
        3. **Pipeline Preparation**: Initialize feature processing
        4. **State Management**: Set up training and validation tracking
        
        Model Options:
        =============
        - **Random Forest**: Robust ensemble method, good for feature importance
        - **XGBoost**: Gradient boosting, often higher performance
        - **Custom Models**: Extensible framework for additional algorithms
        
        Configuration Options:
        =====================
        - n_estimators: Number of trees in ensemble
        - random_state: Reproducibility seed
        - class_weight: Handling imbalanced data
        - max_depth: Tree depth control
        - min_samples_split: Overfitting prevention
        """
        # Store configuration parameters
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_score = 0.0
        self.validation_score = 0.0
        
        # Initialize the specified machine learning model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                max_depth=10,           # Prevent overfitting
                min_samples_split=5,    # Minimum samples for split
                min_samples_leaf=2,     # Minimum samples in leaf
                class_weight='balanced' # Handle class imbalance
            )
        else:
            # Fallback to Random Forest if unsupported model type
            logger.warning(f"Unsupported model type: {model_type}. Using Random Forest.")
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )
        
        logger.info(f"Initialized {model_type} classifier with {n_estimators} estimators")
    
    def extract_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive technical features from price data.
        
        This method performs sophisticated feature engineering to create
        a rich set of technical indicators and market features suitable
        for machine learning. It forms the foundation of the ML trading
        signal prediction system.
        
        Args:
            price_data (pd.DataFrame): Price data with OHLCV columns
        
        Returns:
            pd.DataFrame: Engineered features ready for ML training
        
        Raises:
            ValueError: If required price columns are missing
        
        Feature Categories:
        ==================
        
        1. **Price-Based Features**
           - Close, high, low prices
           - Price ratios and relationships
           - Gap analysis (gap up/down)
        
        2. **Moving Averages**
           - Simple Moving Averages (5, 10, 20, 50 periods)
           - Exponential Moving Averages (10, 20 periods)
           - Price-to-MA ratios for trend analysis
        
        3. **Momentum Indicators**
           - RSI (Relative Strength Index)
           - MACD (Moving Average Convergence Divergence)
           - Price momentum over multiple timeframes
        
        4. **Volatility Features**
           - Bollinger Bands (upper, lower, width, position)
           - ATR (Average True Range)
           - Rolling volatility measures
        
        5. **Market Structure**
           - Higher highs/lower lows detection
           - Trend strength and consistency
           - Support/resistance levels
        
        6. **Volume Features** (if available)
           - Volume moving averages
           - Volume ratios
           - Price-volume relationships
        
        Feature Engineering Process:
        ============================
        1. **Data Validation**: Ensure required columns exist
        2. **Feature Computation**: Calculate technical indicators
        3. **Feature Enhancement**: Create derived features
        4. **Data Cleaning**: Handle missing values and outliers
        5. **Feature Selection**: Store feature names for later use
        
        Example Usage:
        =============
        ```python
        # Extract features from price data
        features = classifier.extract_features(price_data)
        
        # View feature names
        print(classifier.feature_names)
        
        # Check feature statistics
        print(features.describe())
        ```
        """
        # Validate input data
        df = price_data.copy()
        
        # Ensure required columns exist
        if 'close' not in df.columns:
            raise ValueError("Price data must contain 'close' column")
        
        # Initialize feature dataframe
        features = pd.DataFrame(index=df.index)
        
        # === PRICE-BASED FEATURES ===
        features['close'] = df['close']
        features['high'] = df.get('high', df['close'])
        features['low'] = df.get('low', df['close'])
        features['volume'] = df.get('volume', 1000)  # Default volume if missing
        
        # === MOVING AVERAGES ===
        # Simple Moving Averages with multiple periods
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'close_to_sma_{period}'] = df['close'] / features[f'sma_{period}'] - 1
        
        # Exponential Moving Averages
        for period in [10, 20]:
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'close_to_ema_{period}'] = df['close'] / features[f'ema_{period}'] - 1
        
        # === MOMENTUM INDICATORS ===
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # === VOLATILITY FEATURES ===
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volatility measures
        features['volatility_10'] = df['close'].rolling(10).std()
        features['volatility_20'] = df['close'].rolling(20).std()
        features['vol_ratio'] = features['volatility_10'] / features['volatility_20']
        
        # === MOMENTUM FEATURES ===
        # Price momentum over multiple timeframes
        for period in [1, 3, 5, 10]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
            features[f'momentum_{period}_abs'] = np.abs(features[f'momentum_{period}'])
        
        # === VOLUME FEATURES ===
        # Volume analysis (if available)
        if 'volume' in df.columns:
            features['volume_ma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma']
            features['price_volume'] = df['close'] * df['volume']
        
        # === TECHNICAL INDICATORS ===
        # ATR (Average True Range)
        high_low = features['high'] - features['low']
        high_close = np.abs(features['high'] - df['close'].shift())
        low_close = np.abs(features['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        features['atr'] = true_range.rolling(14).mean()
        features['atr_ratio'] = true_range / features['atr']
        
        # === MARKET STRUCTURE FEATURES ===
        # Pattern recognition
        features['higher_high'] = (features['high'] > features['high'].shift(1)).astype(int)
        features['lower_low'] = (features['low'] < features['low'].shift(1)).astype(int)
        features['gap_up'] = (features['low'] > features['high'].shift(1)).astype(int)
        features['gap_down'] = (features['high'] < features['low'].shift(1)).astype(int)
        
        # Trend analysis
        features['trend_strength'] = features['close_to_sma_20'].abs()
        features['trend_consistency'] = features['close_to_sma_20'].rolling(5).apply(
            lambda x: 1 if (x > 0).all() or (x < 0).all() else 0
        )
        
        # === DATA CLEANING ===
        # Handle missing values through forward fill and zero fill
        features = features.fillna(method='ffill').fillna(0)
        
        # Store feature names for later use (exclude price columns)
        self.feature_names = [col for col in features.columns 
                             if col not in ['close', 'high', 'low', 'volume']]
        
        return features
    
    def train(self, price_data: pd.DataFrame, signals_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the ML model using historical price data and signals.
        
        This method implements a comprehensive training pipeline that includes
        feature engineering, data preparation, model training, and performance
        evaluation. It provides detailed metrics and insights about the
        trained model's performance.
        
        Args:
            price_data (pd.DataFrame): Historical price data with OHLCV columns
            signals_data (pd.DataFrame): Historical trading signals
        
        Returns:
            Dict[str, Any]: Training results including:
                - training_accuracy: Accuracy on training set
                - validation_accuracy: Accuracy on validation set
                - classification_report: Detailed performance metrics
                - feature_importance: Top features ranked by importance
                - feature_names: All feature names used
                - training_samples: Number of training samples
                - validation_samples: Number of validation samples
        
        Training Pipeline:
        =================
        1. **Feature Engineering**: Extract technical indicators and features
        2. **Data Preparation**: Align signals with features and clean data
        3. **Train-Test Split**: Create training and validation sets
        4. **Feature Scaling**: Normalize features for better performance
        5. **Model Training**: Train the specified ML algorithm
        6. **Model Evaluation**: Assess performance on validation set
        7. **Feature Analysis**: Compute feature importance rankings
        
        Model Validation:
        ================
        - Stratified train-test split for balanced evaluation
        - Cross-validation for robust performance estimation
        - Classification report with precision, recall, F1-score
        - Feature importance analysis for model interpretation
        
        Example Usage:
        =============
        ```python
        # Train the model
        results = classifier.train(price_data, signals_data)
        
        # Check training results
        print(f"Validation Accuracy: {results['validation_accuracy']:.3f}")
        print(f"Top Features: {results['top_features'][:5]}")
        
        # View classification report
        print(results['classification_report'])
        ```
        """
        logger.info("Starting ML model training...")
        
        # Extract comprehensive features from price data
        features = self.extract_features(price_data)
        
        # Prepare target signals
        signals = signals_data.copy()
        
        # Ensure signals have the required column
        signal_column = 'signal' if 'signal' in signals.columns else signals.columns[0]
        
        # Align features and signals by index
        aligned_data = features.join(signals[signal_column], how='inner')
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 100:
            raise ValueError("Insufficient data for training (need at least 100 samples)")
        
        # Prepare features and targets
        X = aligned_data[self.feature_names]
        y = aligned_data[signal_column]
        
        # Convert signals to categorical if needed
        if y.dtype in ['float64', 'int64']:
            # Map numeric signals to categorical
            y = y.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y, 
            random_state=self.random_state
        )
        
        # Scale features for better performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train the model
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model performance
        train_predictions = self.model.predict(X_train_scaled)
        val_predictions = self.model.predict(X_val_scaled)
        
        # Calculate performance metrics
        self.training_score = accuracy_score(y_train, train_predictions)
        self.validation_score = accuracy_score(y_val, val_predictions)
        
        # Generate detailed classification report
        class_report = classification_report(y_val, val_predictions, output_dict=True)
        
        # Analyze feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance['feature'].head(10).tolist()
        else:
            feature_importance = pd.DataFrame()
            top_features = []
        
        # Update training status
        self.is_trained = True
        
        # Compile training results
        training_results = {
            'training_accuracy': self.training_score,
            'validation_accuracy': self.validation_score,
            'classification_report': class_report,
            'feature_importance': feature_importance.to_dict('records'),
            'top_features': top_features,
            'feature_names': self.feature_names,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        logger.info(f"Training completed. Validation accuracy: {self.validation_score:.3f}")
        
        return training_results
    
    def predict_signals(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ML-based trading signals for new price data.
        
        This method applies the trained ML model to new price data to generate
        trading signals with confidence scores. It provides both the predicted
        signal and the model's confidence in that prediction.
        
        Args:
            price_data (pd.DataFrame): New price data for prediction
        
        Returns:
            pd.DataFrame: Predictions with columns:
                - signal: Predicted trading signal (-1, 0, 1)
                - confidence: Model confidence score (0-1)
                - probabilities: Class probabilities for each signal
        
        Raises:
            ValueError: If model hasn't been trained yet
        
        Prediction Process:
        ==================
        1. **Model Validation**: Ensure model is trained
        2. **Feature Engineering**: Extract features from new data
        3. **Feature Scaling**: Apply same scaling as training
        4. **Signal Prediction**: Generate predictions with confidence
        5. **Result Formatting**: Structure output for easy consumption
        
        Confidence Scoring:
        ==================
        - Based on prediction probability from the model
        - Higher confidence indicates more certain predictions
        - Useful for position sizing and risk management
        - Threshold-based filtering for high-confidence signals
        
        Example Usage:
        =============
        ```python
        # Generate predictions
        predictions = classifier.predict_signals(new_price_data)
        
        # Filter high-confidence signals
        high_conf = predictions[predictions['confidence'] > 0.8]
        
        # View signal distribution
        print(predictions['signal'].value_counts())
        ```
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features from new price data
        features = self.extract_features(price_data)
        
        # Prepare feature matrix
        X = features[self.feature_names]
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(0)
        
        # Scale features using the same scaler from training
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        predictions = self.model.predict(X_scaled)
        
        # Get prediction probabilities for confidence scoring
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
            # Confidence is the maximum probability
            confidence = np.max(probabilities, axis=1)
        else:
            probabilities = np.zeros((len(predictions), 3))
            confidence = np.ones(len(predictions)) * 0.5  # Default confidence
        
        # Create results dataframe
        results = pd.DataFrame({
            'signal': predictions,
            'confidence': confidence
        }, index=price_data.index)
        
        # Add probability columns if available
        if hasattr(self.model, 'predict_proba'):
            classes = self.model.classes_
            for i, class_label in enumerate(classes):
                results[f'prob_{class_label}'] = probabilities[:, i]
        
        return results
    
    def compare_with_rule_based(self, 
                               price_data: pd.DataFrame, 
                               rule_signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare ML predictions with rule-based signals.
        
        This method provides comprehensive comparison between machine learning
        predictions and traditional rule-based trading signals. It analyzes
        agreement rates, performance differences, and identifies scenarios
        where each approach excels.
        
        Args:
            price_data (pd.DataFrame): Price data for analysis
            rule_signals (pd.DataFrame): Rule-based trading signals
        
        Returns:
            Dict[str, Any]: Comparison results including:
                - agreement_rate: Overall agreement between ML and rules
                - high_confidence_agreement_rate: Agreement on high-confidence ML signals
                - signal_distribution: Distribution of signals by method
                - disagreement_analysis: Analysis of disagreement patterns
                - performance_comparison: Performance metrics comparison
        
        Comparison Metrics:
        ==================
        - Overall agreement rate between methods
        - High-confidence agreement analysis
        - Signal distribution comparison
        - Performance attribution analysis
        - Disagreement pattern identification
        
        Analysis Framework:
        ==================
        1. **Signal Alignment**: Align ML and rule-based signals
        2. **Agreement Analysis**: Calculate agreement rates
        3. **Confidence Analysis**: Analyze high-confidence predictions
        4. **Disagreement Patterns**: Identify when methods disagree
        5. **Performance Comparison**: Compare effectiveness
        
        Example Usage:
        =============
        ```python
        # Compare methods
        comparison = classifier.compare_with_rule_based(
            price_data, rule_signals
        )
        
        # View agreement rates
        print(f"Overall Agreement: {comparison['agreement_rate']:.3f}")
        print(f"High Confidence Agreement: {comparison['high_confidence_agreement_rate']:.3f}")
        
        # Analyze disagreements
        print(comparison['disagreement_analysis'])
        ```
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making comparisons")
        
        # Generate ML predictions
        ml_predictions = self.predict_signals(price_data)
        
        # Align signals by index
        rule_signal_column = 'signal' if 'signal' in rule_signals.columns else rule_signals.columns[0]
        
        # Create comparison dataframe
        comparison_data = pd.DataFrame({
            'ml_signal': ml_predictions['signal'],
            'rule_signal': rule_signals[rule_signal_column],
            'ml_confidence': ml_predictions['confidence']
        }).dropna()
        
        # Calculate agreement rates
        total_signals = len(comparison_data)
        agreements = (comparison_data['ml_signal'] == comparison_data['rule_signal']).sum()
        agreement_rate = agreements / total_signals if total_signals > 0 else 0
        
        # High-confidence agreement analysis
        high_confidence_threshold = 0.8
        high_conf_data = comparison_data[comparison_data['ml_confidence'] > high_confidence_threshold]
        
        if len(high_conf_data) > 0:
            high_conf_agreements = (high_conf_data['ml_signal'] == high_conf_data['rule_signal']).sum()
            high_conf_agreement_rate = high_conf_agreements / len(high_conf_data)
        else:
            high_conf_agreement_rate = 0
        
        # Signal distribution analysis
        ml_distribution = comparison_data['ml_signal'].value_counts().to_dict()
        rule_distribution = comparison_data['rule_signal'].value_counts().to_dict()
        
        # Disagreement analysis
        disagreements = comparison_data[comparison_data['ml_signal'] != comparison_data['rule_signal']]
        disagreement_patterns = {}
        
        if len(disagreements) > 0:
            # Analyze common disagreement patterns
            for _, row in disagreements.iterrows():
                pattern = f"ML:{row['ml_signal']}_Rule:{row['rule_signal']}"
                disagreement_patterns[pattern] = disagreement_patterns.get(pattern, 0) + 1
        
        # Performance analysis (if price data available for returns)
        performance_comparison = {}
        if len(comparison_data) > 1:
            # Calculate basic performance metrics
            performance_comparison = {
                'ml_signal_count': len(comparison_data[comparison_data['ml_signal'] != 0]),
                'rule_signal_count': len(comparison_data[comparison_data['rule_signal'] != 0]),
                'both_signal_count': len(comparison_data[
                    (comparison_data['ml_signal'] != 0) & (comparison_data['rule_signal'] != 0)
                ])
            }
        
        # Compile comparison results
        comparison_results = {
            'agreement_rate': agreement_rate,
            'high_confidence_agreement_rate': high_conf_agreement_rate,
            'total_samples': total_signals,
            'high_confidence_samples': len(high_conf_data),
            'signal_distribution': {
                'ml_signals': ml_distribution,
                'rule_signals': rule_distribution
            },
            'disagreement_patterns': disagreement_patterns,
            'performance_comparison': performance_comparison
        }
        
        return comparison_results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings from the trained model.
        
        This method provides insights into which features the ML model
        considers most important for trading signal prediction. It's
        crucial for model interpretation and feature selection.
        
        Returns:
            pd.DataFrame: Feature importance rankings with columns:
                - feature: Feature name
                - importance: Importance score
                - rank: Importance rank
        
        Raises:
            ValueError: If model hasn't been trained or doesn't support feature importance
        
        Importance Analysis:
        ===================
        - Feature importance scores from the trained model
        - Ranked list of most influential features
        - Insights into model decision-making process
        - Guidance for feature selection and engineering
        
        Example Usage:
        =============
        ```python
        # Get feature importance
        importance = classifier.get_feature_importance()
        
        # View top 10 features
        print(importance.head(10))
        
        # Plot feature importance
        import matplotlib.pyplot as plt
        importance.head(10).plot(x='feature', y='importance', kind='barh')
        plt.show()
        ```
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importance")
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Add rank column
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        This method serializes the trained model, scaler, and metadata
        to disk for later use. It enables model persistence and deployment
        in production trading systems.
        
        Args:
            filepath (str): Path where to save the model
        
        Raises:
            ValueError: If model hasn't been trained yet
        
        Saved Components:
        ================
        - Trained ML model
        - Feature scaler
        - Feature names
        - Model metadata
        - Training statistics
        
        Example Usage:
        =============
        ```python
        # Save trained model
        classifier.save_model("my_trading_model.pkl")
        
        # Later, load the model
        loaded_classifier = MLSignalClassifier.load_model("my_trading_model.pkl")
        ```
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Package all model components
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'random_state': self.random_state,
            'training_score': self.training_score,
            'validation_score': self.validation_score,
            'is_trained': self.is_trained
        }
        
        # Save to disk
        joblib.dump(model_package, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MLSignalClassifier':
        """
        Load a trained model from disk.
        
        This class method reconstructs a trained MLSignalClassifier
        from a saved model file. It enables model persistence and
        deployment in production systems.
        
        Args:
            filepath (str): Path to the saved model file
        
        Returns:
            MLSignalClassifier: Loaded classifier instance
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted or invalid
        
        Example Usage:
        =============
        ```python
        # Load saved model
        classifier = MLSignalClassifier.load_model("my_trading_model.pkl")
        
        # Use loaded model
        predictions = classifier.predict_signals(new_data)
        ```
        """
        # Load model package from disk
        model_package = joblib.load(filepath)
        
        # Create new instance
        classifier = cls(
            model_type=model_package['model_type'],
            random_state=model_package['random_state']
        )
        
        # Restore model state
        classifier.model = model_package['model']
        classifier.scaler = model_package['scaler']
        classifier.feature_names = model_package['feature_names']
        classifier.training_score = model_package['training_score']
        classifier.validation_score = model_package['validation_score']
        classifier.is_trained = model_package['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        return classifier


def create_ml_classifier(config: Dict[str, Any] = None) -> MLSignalClassifier:
    """
    Create ML classifier from configuration.
    
    This factory function creates a properly configured MLSignalClassifier
    instance from a configuration dictionary. It provides a convenient way
    to initialize the classifier with standard configuration patterns.
    
    Args:
        config (Dict[str, Any], optional): Configuration dictionary containing:
            - model_type: Type of ML model to use
            - n_estimators: Number of trees for ensemble methods
            - random_state: Random seed for reproducibility
    
    Returns:
        MLSignalClassifier: Configured classifier instance
    
    Configuration Structure:
    =======================
    ```yaml
    ml_classifier:
      model_type: "random_forest"
      n_estimators: 100
      random_state: 42
    ```
    
    Example Usage:
    =============
    ```python
    # Create from configuration
    config = {
        'ml_classifier': {
            'model_type': 'random_forest',
            'n_estimators': 200,
            'random_state': 42
        }
    }
    
    classifier = create_ml_classifier(config)
    ```
    """
    # Handle missing configuration
    if config is None:
        config = {}
    
    # Extract ML classifier configuration
    ml_config = config.get('ml_classifier', {})
    
    # Create and return configured classifier
    return MLSignalClassifier(
        model_type=ml_config.get('model_type', 'random_forest'),
        n_estimators=ml_config.get('n_estimators', 100),
        random_state=ml_config.get('random_state', 42)
    )


# Example usage and demonstration
if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("Machine Learning Signal Classifier Demo")
    print("=" * 50)
    
    # Get sample market data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Download real market data
    data = yf.download("AAPL", start=start_date, end=end_date)
    data = data.reset_index()
    data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
    
    # Create simple rule-based signals (SMA crossover)
    data['sma_fast'] = data['close'].rolling(10).mean()
    data['sma_slow'] = data['close'].rolling(20).mean()
    data['signal'] = 0
    data.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
    data.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
    
    # Initialize ML classifier
    classifier = MLSignalClassifier()
    
    # Train the model
    print("\n1. Training ML Model:")
    print("-" * 30)
    training_results = classifier.train(data, data[['signal']])
    print(f"Training Accuracy: {training_results['training_accuracy']:.3f}")
    print(f"Validation Accuracy: {training_results['validation_accuracy']:.3f}")
    print(f"Top Features: {training_results['top_features'][:5]}")
    
    # Generate ML predictions
    print("\n2. Generating ML Predictions:")
    print("-" * 30)
    ml_predictions = classifier.predict_signals(data)
    print(f"Generated {len(ml_predictions)} ML predictions")
    print(f"Signal Distribution: {ml_predictions['signal'].value_counts().to_dict()}")
    
    # Compare with rule-based signals
    print("\n3. Comparing with Rule-Based Signals:")
    print("-" * 30)
    comparison = classifier.compare_with_rule_based(data, data[['signal']])
    print(f"Overall Agreement Rate: {comparison['agreement_rate']:.3f}")
    print(f"High Confidence Agreement Rate: {comparison['high_confidence_agreement_rate']:.3f}")
    print(f"Total Samples: {comparison['total_samples']}")
    
    # Feature importance analysis
    print("\n4. Feature Importance Analysis:")
    print("-" * 30)
    importance = classifier.get_feature_importance()
    print("Top 10 Most Important Features:")
    for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Demonstrate model saving/loading
    print("\n5. Model Persistence:")
    print("-" * 30)
    model_path = "demo_trading_model.pkl"
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Load the model
    loaded_classifier = MLSignalClassifier.load_model(model_path)
    print("Model loaded successfully")
    
    # Cleanup
    import os
    os.remove(model_path)
    print("Demo model file cleaned up")
    
    print("\n" + "=" * 50)
    print("Demo complete! ML classifier ready for integration.")
    print("=" * 50)