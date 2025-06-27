"""Machine Learning signal classifier for comparing ML vs rule-based strategies."""

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
    """Machine learning classifier to predict trading signals."""
    
    def __init__(self, 
                 model_type: str = "random_forest",
                 n_estimators: int = 100,
                 random_state: int = 42):
        """Initialize ML classifier.
        
        Args:
            model_type: Type of ML model ('random_forest', 'xgboost')
            n_estimators: Number of trees for ensemble methods
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_score = 0.0
        self.validation_score = 0.0
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'
            )
        else:
            # Fallback to RandomForest
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )
        
        logger.info(f"Initialized {model_type} classifier")
    
    def extract_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical features for ML training."""
        
        df = price_data.copy()
        
        # Ensure we have the basic columns
        if 'close' not in df.columns:
            raise ValueError("Price data must contain 'close' column")
        
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['close'] = df['close']
        features['high'] = df.get('high', df['close'])
        features['low'] = df.get('low', df['close'])
        features['volume'] = df.get('volume', 1000)  # Default volume if missing
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'close_to_sma_{period}'] = df['close'] / features[f'sma_{period}'] - 1
        
        # EMA
        for period in [10, 20]:
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'close_to_ema_{period}'] = df['close'] / features[f'ema_{period}'] - 1
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volatility features
        features['volatility_10'] = df['close'].rolling(10).std()
        features['volatility_20'] = df['close'].rolling(20).std()
        features['vol_ratio'] = features['volatility_10'] / features['volatility_20']
        
        # Price momentum
        for period in [1, 3, 5, 10]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
            features[f'momentum_{period}_abs'] = np.abs(features[f'momentum_{period}'])
        
        # Volume features (if available)
        if 'volume' in df.columns:
            features['volume_ma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma']
            features['price_volume'] = df['close'] * df['volume']
        
        # ATR (Average True Range)
        high_low = features['high'] - features['low']
        high_close = np.abs(features['high'] - df['close'].shift())
        low_close = np.abs(features['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        features['atr'] = true_range.rolling(14).mean()
        features['atr_ratio'] = true_range / features['atr']
        
        # Market structure features
        features['higher_high'] = (features['high'] > features['high'].shift(1)).astype(int)
        features['lower_low'] = (features['low'] < features['low'].shift(1)).astype(int)
        features['gap_up'] = (features['low'] > features['high'].shift(1)).astype(int)
        features['gap_down'] = (features['high'] < features['low'].shift(1)).astype(int)
        
        # Trend strength
        features['trend_strength'] = features['close_to_sma_20'].abs()
        features['trend_consistency'] = features['close_to_sma_20'].rolling(5).apply(
            lambda x: 1 if (x > 0).all() or (x < 0).all() else 0
        )
        
        # Clean features
        features = features.fillna(method='ffill').fillna(0)
        
        # Store feature names
        self.feature_names = [col for col in features.columns if col not in ['close', 'high', 'low', 'volume']]
        
        return features[self.feature_names]
    
    def prepare_training_data(self, 
                            price_data: pd.DataFrame, 
                            signals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training."""
        
        # Extract features
        features = self.extract_features(price_data)
        
        # Get signals (assuming 'signal' column with 1, 0, -1)
        if 'signal' not in signals.columns:
            raise ValueError("Signals dataframe must contain 'signal' column")
        
        # Align features and signals
        aligned_data = pd.concat([features, signals['signal']], axis=1, join='inner')
        aligned_data = aligned_data.dropna()
        
        # Convert signals to classification labels
        # 1 = Buy, 0 = Hold, -1 = Sell → 2, 1, 0 for sklearn
        labels = aligned_data['signal'].map({1: 2, 0: 1, -1: 0})
        
        features_clean = aligned_data[self.feature_names]
        
        logger.info(f"Prepared {len(features_clean)} samples with {len(self.feature_names)} features")
        logger.info(f"Signal distribution: {labels.value_counts().to_dict()}")
        
        return features_clean, labels
    
    def train(self, 
              price_data: pd.DataFrame, 
              rule_based_signals: pd.DataFrame,
              test_size: float = 0.2) -> Dict[str, Any]:
        """Train the ML model on rule-based signals."""
        
        # Prepare data
        X, y = self.prepare_training_data(price_data, rule_based_signals)
        
        if len(X) < 50:
            raise ValueError("Not enough data for training (minimum 50 samples required)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info("Training ML model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        self.training_score = accuracy_score(y_train, train_pred)
        self.validation_score = accuracy_score(y_test, test_pred)
        self.is_trained = True
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        training_results = {
            'training_accuracy': self.training_score,
            'validation_accuracy': self.validation_score,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(self.feature_names),
            'top_features': top_features,
            'classification_report': classification_report(y_test, test_pred, output_dict=True)
        }
        
        logger.info(f"Training complete. Validation accuracy: {self.validation_score:.3f}")
        
        return training_results
    
    def predict_signals(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate ML-based signals for new data."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_features(price_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features[self.feature_names])
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Convert back to signal format (2,1,0 → 1,0,-1)
        signal_map = {2: 1, 1: 0, 0: -1}
        signals = pd.Series(predictions, index=features.index).map(signal_map)
        
        # Create results dataframe
        results = pd.DataFrame(index=features.index)
        results['ml_signal'] = signals
        results['ml_confidence'] = probabilities.max(axis=1)  # Highest probability
        
        # Add probability breakdown
        results['prob_buy'] = probabilities[:, 2] if probabilities.shape[1] > 2 else 0
        results['prob_hold'] = probabilities[:, 1] if probabilities.shape[1] > 1 else 0
        results['prob_sell'] = probabilities[:, 0] if probabilities.shape[1] > 0 else 0
        
        return results
    
    def compare_with_rule_based(self, 
                              price_data: pd.DataFrame,
                              rule_signals: pd.DataFrame) -> Dict[str, Any]:
        """Compare ML signals with rule-based signals."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before comparison")
        
        # Get ML predictions
        ml_results = self.predict_signals(price_data)
        
        # Align data
        comparison_data = pd.concat([
            rule_signals[['signal']].rename(columns={'signal': 'rule_signal'}),
            ml_results[['ml_signal', 'ml_confidence']]
        ], axis=1, join='inner')
        
        comparison_data = comparison_data.dropna()
        
        # Calculate agreement metrics
        total_signals = len(comparison_data)
        agreements = (comparison_data['rule_signal'] == comparison_data['ml_signal']).sum()
        agreement_rate = agreements / total_signals if total_signals > 0 else 0
        
        # Signal distribution comparison
        rule_dist = comparison_data['rule_signal'].value_counts(normalize=True).to_dict()
        ml_dist = comparison_data['ml_signal'].value_counts(normalize=True).to_dict()
        
        # Confidence analysis
        high_confidence_threshold = 0.7
        high_confidence_signals = comparison_data[comparison_data['ml_confidence'] > high_confidence_threshold]
        high_conf_agreement = (
            (high_confidence_signals['rule_signal'] == high_confidence_signals['ml_signal']).sum() /
            len(high_confidence_signals)
        ) if len(high_confidence_signals) > 0 else 0
        
        comparison_results = {
            'total_signals': total_signals,
            'agreement_rate': agreement_rate,
            'agreements': agreements,
            'disagreements': total_signals - agreements,
            'rule_signal_distribution': rule_dist,
            'ml_signal_distribution': ml_dist,
            'high_confidence_signals': len(high_confidence_signals),
            'high_confidence_agreement_rate': high_conf_agreement,
            'average_ml_confidence': comparison_data['ml_confidence'].mean()
        }
        
        return comparison_results
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'training_score': self.training_score,
            'validation_score': self.validation_score,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.training_score = model_data['training_score']
        self.validation_score = model_data['validation_score']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Model type: {self.model_type}, Validation score: {self.validation_score:.3f}")


def create_ml_classifier(config: Dict[str, Any] = None) -> MLSignalClassifier:
    """Factory function to create ML classifier from config."""
    
    if config is None:
        config = {}
    
    ml_config = config.get('ml_classifier', {})
    
    return MLSignalClassifier(
        model_type=ml_config.get('model_type', 'random_forest'),
        n_estimators=ml_config.get('n_estimators', 100),
        random_state=ml_config.get('random_state', 42)
    )


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Get sample data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = yf.download("AAPL", start=start_date, end=end_date)
    data = data.reset_index()
    data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
    
    # Create simple rule-based signals
    data['sma_fast'] = data['close'].rolling(10).mean()
    data['sma_slow'] = data['close'].rolling(20).mean()
    data['signal'] = 0
    data.loc[data['sma_fast'] > data['sma_slow'], 'signal'] = 1
    data.loc[data['sma_fast'] < data['sma_slow'], 'signal'] = -1
    
    # Initialize and train ML classifier
    classifier = MLSignalClassifier()
    
    # Train model
    training_results = classifier.train(data, data[['signal']])
    print("Training Results:")
    print(f"Validation Accuracy: {training_results['validation_accuracy']:.3f}")
    print(f"Top Features: {training_results['top_features'][:5]}")
    
    # Generate ML predictions
    ml_predictions = classifier.predict_signals(data)
    print(f"\nGenerated {len(ml_predictions)} ML signals")
    
    # Compare with rule-based
    comparison = classifier.compare_with_rule_based(data, data[['signal']])
    print(f"\nComparison Results:")
    print(f"Agreement Rate: {comparison['agreement_rate']:.3f}")
    print(f"High Confidence Agreement: {comparison['high_confidence_agreement_rate']:.3f}")