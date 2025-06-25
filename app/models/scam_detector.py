import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
import os 
import re
import string
from collections import Counter
import json

class ScamDetector: 
    def __init__(self): 
        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english')) 
        except:
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet')
            self.stop_words = set(stopwords.words('english'))
        
        self.lemmatizer = WordNetLemmatizer() 
        
        # TF-IDF vectorizer for main feature extraction
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8) 
        
        # N-gram vectorizers for pattern analysis
        self.bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=1000)
        self.trigram_vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=500)
        
        # Main classification model with better parameters
        self.model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'
        )
        
        # Platform classification model
        self.platform_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=8,
            class_weight='balanced'
        )
        
        # Scam type classification model (NEW)
        self.scam_type_model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=8,
            class_weight='balanced'
        )
        
        # Platform vectorizer
        self.platform_vectorizer = TfidfVectorizer(max_features=3000, min_df=1, max_df=0.9)
        
        # Scam type vectorizer (NEW)
        self.scam_type_vectorizer = TfidfVectorizer(max_features=3000, min_df=1, max_df=0.9)
        
        # Initialize data files
        self.ensure_data_files_exist() 
        
        # Load training data with error handling
        try:
            self.train_data = pd.read_csv('data/training_data.csv')
            print(f"Loaded {len(self.train_data)} training samples")
        except (pd.errors.ParserError, FileNotFoundError) as e:
            print(f"Error loading training data: {e}")
            print("Recreating training data file...")
            self.recreate_training_data()
            self.train_data = pd.read_csv('data/training_data.csv')
        
        # Load or initialize suspicious n-grams
        self.suspicious_ngrams_file = 'data/suspicious_ngrams.json'
        self.load_suspicious_ngrams()
        
        if len(self.train_data) < 2: 
            self.add_sample_data() 
        
        # Train the model with initial data
        self.train_model()
    
    def load_suspicious_ngrams(self):
        """Load suspicious n-grams from file or initialize with defaults"""
        if os.path.exists(self.suspicious_ngrams_file):
            try:
                with open(self.suspicious_ngrams_file, 'r') as f:
                    ngrams_data = json.load(f)
                    self.suspicious_bigrams = ngrams_data.get('bigrams', {})
                    self.suspicious_trigrams = ngrams_data.get('trigrams', {})
                print(f"Loaded {len(self.suspicious_bigrams)} bigrams and {len(self.suspicious_trigrams)} trigrams")
            except Exception as e:
                print(f"Error loading suspicious n-grams: {e}")
                self.initialize_default_ngrams()
        else:
            self.initialize_default_ngrams()
    
    def save_suspicious_ngrams(self):
        """Save suspicious n-grams to file"""
        ngrams_data = {
            'bigrams': self.suspicious_bigrams,
            'trigrams': self.suspicious_trigrams
        }
        os.makedirs(os.path.dirname(self.suspicious_ngrams_file), exist_ok=True)
        with open(self.suspicious_ngrams_file, 'w') as f:
            json.dump(ngrams_data, f)
        print(f"Saved {len(self.suspicious_bigrams)} bigrams and {len(self.suspicious_trigrams)} trigrams")
    
    def initialize_default_ngrams(self):
        """Initialize default suspicious n-grams with higher thresholds"""
        # Define common suspicious n-grams in scams with their suspicion scores
        self.suspicious_bigrams = {
            'click here': 0.95,
            'verify account': 0.9,
            'bank details': 0.85,
            'urgent action': 0.95,
            'limited time': 0.85,
            'send money': 0.98,
            'won lottery': 0.95,
            'claim prize': 0.9,
            'password to': 0.95,
            'free iphone': 0.9,
            'gift card': 0.8,
            'personal information': 0.85,
            'credit card': 0.8,
            'social security': 0.95,
            'act now': 0.9,
            'verify your': 0.8,
            'confirm your': 0.8,
            'please verify': 0.85,
            'please confirm': 0.85,
            'security alert': 0.9,
            'account suspended': 0.9,
            'account locked': 0.9,
            'unusual activity': 0.85,
            'suspicious activity': 0.9,
            'identity verification': 0.85,
            'verify identity': 0.85,
            'unauthorized access': 0.95,
            'click link': 0.9,
            'click below': 0.85,
            'update your': 0.75,
            'security reasons': 0.8,
            'account information': 0.75,
            'card details': 0.9,
            'card number': 0.95,
            'cvv number': 0.98,
            'pin number': 0.95,
            'wire transfer': 0.9,
            'money transfer': 0.9,
            'bitcoin payment': 0.95,
            'crypto payment': 0.95,
            'final notice': 0.85,
            'last warning': 0.95,
            'account termination': 0.85,
            'legal action': 0.85,
            'tax refund': 0.85,
            'irs notice': 0.9,
            'prize winning': 0.95,
            'lottery winning': 0.98,
            'cash prize': 0.9
        }
        
        self.suspicious_trigrams = {
            'verify your account': 0.95,
            'send bank details': 0.98,
            'click this link': 0.95,
            'won a prize': 0.9,
            'claim your prize': 0.9,
            'urgent action required': 0.98,
            'limited time offer': 0.9,
            'send us your': 0.95,
            'has been compromised': 0.98,
            'need your password': 0.98,
            'provide your details': 0.9,
            'confirm your identity': 0.85,
            'suspicious activity detected': 0.95,
            'unusual activity detected': 0.95,
            'click the link': 0.9,
            'please verify your': 0.9,
            'please confirm your': 0.9,
            'for security reasons': 0.85,
            'enter your password': 0.95,
            'enter your pin': 0.98,
            'enter your cvv': 0.98,
            'won the lottery': 0.98,
            'selected as winner': 0.95,
            'transfer the money': 0.95,
            'send the money': 0.95,
            'legal action will': 0.95,
            'account will terminate': 0.9,
            'has been suspended': 0.9,
            'due to suspicious': 0.95
        }
        
        # Save the default n-grams
        self.save_suspicious_ngrams()
 
    def ensure_data_files_exist(self): 
        os.makedirs('data', exist_ok=True) 
        files = { 
            'data/training_data.csv': ['text', 'is_scam', 'scam_type', 'platform'], 
            'data/flagged_messages.csv': ['text', 'is_scam', 'scam_type', 'platform'] 
        } 
        for file, columns in files.items(): 
            if not os.path.exists(file): 
                pd.DataFrame(columns=columns).to_csv(file, index=False) 
                print(f"Created {file}")

    def recreate_training_data(self):
        """Recreate the training data file if it's corrupted"""
        print("Recreating training_data.csv...")
        
        # Create a fresh training data file
        columns = ['text', 'is_scam', 'scam_type', 'platform']
        fresh_data = pd.DataFrame(columns=columns)
        fresh_data.to_csv('data/training_data.csv', index=False)
        
        # Try to salvage data from flagged_messages.csv if it exists
        try:
            if os.path.exists('data/flagged_messages.csv'):
                flagged_data = pd.read_csv('data/flagged_messages.csv')
                if len(flagged_data) > 0:
                    # Ensure the flagged data has the right columns
                    if all(col in flagged_data.columns for col in columns):
                        flagged_data.to_csv('data/training_data.csv', index=False)
                        print(f"Restored {len(flagged_data)} samples from flagged_messages.csv")
                    else:
                        print("Flagged messages file has incorrect format, starting fresh")
        except Exception as e:
            print(f"Could not restore from flagged messages: {e}")
        
        print("Training data file recreated successfully")
 
    def add_sample_data(self): 
        sample_data = [ 
            { 
                "text": "Congratulations! You've won a free iPhone. Click here to claim your prize!", 
                "is_scam": True, 
                "scam_type": "phishing", 
                "platform": "telegram" 
            }, 
            { 
                "text": "Hi, how are you doing? I hope you're having a great day!", 
                "is_scam": False, 
                "scam_type": "none", 
                "platform": "whatsapp" 
            },
            {
                "text": "URGENT: Your account has been compromised. Click here to verify your identity immediately!",
                "is_scam": True,
                "scam_type": "urgent",
                "platform": "telegram"
            },
            {
                "text": "This is your bank. We need you to confirm your account details by sending your password.",
                "is_scam": True,
                "scam_type": "impersonation",
                "platform": "whatsapp"
            },
            {
                "text": "Hey, I'm having a great time at the conference. Will call you later tonight.",
                "is_scam": False,
                "scam_type": "none",
                "platform": "telegram"
            },
            {
                "text": "You have won a lottery! Send us your bank details to claim your prize money immediately.",
                "is_scam": True,
                "scam_type": "financial",
                "platform": "telegram"
            },
            {
                "text": "Your PayPal account has been limited. Click here to verify your information and restore full access.",
                "is_scam": True,
                "scam_type": "phishing",
                "platform": "telegram"
            },
            {
                "text": "Thanks for the meeting today. Let's schedule a follow-up next week.",
                "is_scam": False,
                "scam_type": "none",
                "platform": "telegram"
            },
            {
                "text": "Happy birthday! Hope you have a wonderful day.",
                "is_scam": False,
                "scam_type": "none",
                "platform": "whatsapp"
            },
            {
                "text": "Can you pick up some groceries on your way home?",
                "is_scam": False,
                "scam_type": "none",
                "platform": "whatsapp"
            },
            {
                "text": "Meeting is rescheduled to 3 PM tomorrow.",
                "is_scam": False,
                "scam_type": "none",
                "platform": "telegram"
            },
            {
                "text": "Your package delivery failed. Click here to reschedule.",
                "is_scam": True,
                "scam_type": "phishing",
                "platform": "whatsapp"
            },
            {
                "text": "Final notice: Your subscription will expire. Update payment details now.",
                "is_scam": True,
                "scam_type": "urgent",
                "platform": "telegram"
            },
            {
                "text": "Good morning! How was your weekend?",
                "is_scam": False,
                "scam_type": "none",
                "platform": "whatsapp"
            },
            {
                "text": "Reminder: Doctor appointment tomorrow at 2 PM.",
                "is_scam": False,
                "scam_type": "none",
                "platform": "whatsapp"
            }
        ] 
        self.train_data = pd.concat([self.train_data, pd.DataFrame(sample_data)], ignore_index=True) 
        self.train_data.to_csv('data/training_data.csv', index=False) 
        print("Added sample data to training_data.csv") 
 
    def clean_text(self, text):
        """
        Clean text by removing special characters, numbers, and extra whitespace
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text): 
        """
        Preprocess text for TF-IDF vectorization
        Returns both the cleaned text and the processed tokens for analysis
        """
        # First clean the text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Join tokens back into a string for vectorization
        processed_text = ' '.join(processed_tokens)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'processed_text': processed_text,
            'tokens': processed_tokens
        }
    
    def extract_ngrams(self, text, n=2):
        """
        Extract n-grams from text
        """
        # Clean and tokenize the text
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        # Generate n-grams
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def get_suspicious_ngrams(self, text):
        """
        Extract and score suspicious n-grams from text with improved scoring
        """
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Get all bigrams and trigrams from the text
        text_bigrams = self.extract_ngrams(text_lower, 2)
        text_trigrams = self.extract_ngrams(text_lower, 3)
        
        found_bigrams = []
        found_trigrams = []
        
        # Debug information
        debug_info = {
            'text_length': len(text),
            'text_bigrams': len(text_bigrams),
            'text_trigrams': len(text_trigrams),
            'suspicious_bigrams': len(self.suspicious_bigrams),
            'suspicious_trigrams': len(self.suspicious_trigrams),
            'matches': []
        }
        
        # Check for suspicious bigrams with exact matching only
        for text_bigram in text_bigrams:
            if text_bigram in self.suspicious_bigrams:
                score = self.suspicious_bigrams[text_bigram]
                found_bigrams.append({
                    'ngram': text_bigram,
                    'score': score,
                    'type': 'bigram',
                    'match_type': 'exact'
                })
                debug_info['matches'].append(f"Exact bigram match: {text_bigram}")
        
        # Check for suspicious trigrams with exact matching only
        for text_trigram in text_trigrams:
            if text_trigram in self.suspicious_trigrams:
                score = self.suspicious_trigrams[text_trigram]
                found_trigrams.append({
                    'ngram': text_trigram,
                    'score': score,
                    'type': 'trigram',
                    'match_type': 'exact'
                })
                debug_info['matches'].append(f"Exact trigram match: {text_trigram}")
        
        # Combine and sort by score
        all_ngrams = found_bigrams + found_trigrams
        all_ngrams.sort(key=lambda x: x['score'], reverse=True)
        
        # Add debug information
        debug_info['found_ngrams'] = len(all_ngrams)
        
        return {
            'ngrams': all_ngrams,
            'debug': debug_info
        }
 
    def train_model(self): 
        if len(self.train_data) > 1: 
            # Process all texts in the training data
            processed_data = [self.preprocess_text(text)['processed_text'] for text in self.train_data['text']]
            
            # Fit TF-IDF vectorizer and transform the processed texts
            X_tfidf = self.vectorizer.fit_transform(processed_data)
            
            # Get the target variable for scam detection
            y = self.train_data['is_scam']
            
            # Train the scam detection model
            self.model.fit(X_tfidf, y)
            print("Scam detection model trained successfully with", len(self.train_data), "samples")
            
            # Train platform prediction model if we have enough data
            if len(self.train_data['platform'].unique()) > 1:
                # Fit platform vectorizer
                X_platform = self.platform_vectorizer.fit_transform(processed_data)
                
                # Get platform labels
                y_platform = self.train_data['platform']
                
                # Train platform model
                self.platform_model.fit(X_platform, y_platform)
                print("Platform prediction model trained successfully")
            else:
                print("Not enough platform variety to train platform model")
            
            # Train scam type prediction model (NEW)
            # Only train on scam messages (not legitimate ones)
            scam_data = self.train_data[self.train_data['is_scam'] == True]
            if len(scam_data) > 1 and len(scam_data['scam_type'].unique()) > 1:
                # Get processed text for scam messages only
                scam_processed_data = [self.preprocess_text(text)['processed_text'] for text in scam_data['text']]
                
                # Fit scam type vectorizer
                X_scam_type = self.scam_type_vectorizer.fit_transform(scam_processed_data)
                
                # Get scam type labels (excluding 'none' and 'legitimate')
                y_scam_type = scam_data['scam_type']
                
                # Train scam type model
                self.scam_type_model.fit(X_scam_type, y_scam_type)
                print(f"Scam type prediction model trained successfully with {len(scam_data)} scam samples")
                print(f"Scam types in training data: {scam_data['scam_type'].unique()}")
            else:
                print("Not enough scam type variety to train scam type model")
        else: 
            print("Insufficient data to train the model. Please add more data.") 
 
    def predict(self, message): 
        if len(self.train_data) < 2: 
            return { 
                'risk_level': 'Unknown', 
                'confidence': 0, 
                'message': 'Insufficient training data. Please add more data and retrain the model.' 
            } 
        
        # Process the text and get all the components
        processed_data = self.preprocess_text(message)
        processed_text = processed_data['processed_text']
        
        # Get suspicious n-grams first
        suspicious_ngrams_result = self.get_suspicious_ngrams(message)
        suspicious_ngrams = suspicious_ngrams_result['ngrams']
        debug_info = suspicious_ngrams_result['debug']
        
        # Calculate n-gram based confidence
        ngram_confidence = 0
        if suspicious_ngrams:
            # Use the highest scoring n-gram as primary indicator
            max_ngram_score = max([ngram['score'] for ngram in suspicious_ngrams])
            # Weight by number of suspicious n-grams found
            ngram_weight = min(len(suspicious_ngrams) / 3.0, 1.0)  # Cap at 1.0
            ngram_confidence = max_ngram_score * ngram_weight * 100
        
        # Vectorize the processed text for ML model
        X = self.vectorizer.transform([processed_text]) 
        
        # Get ML model prediction and confidence
        ml_prediction = self.model.predict(X)[0] 
        ml_confidence = self.model.predict_proba(X)[0][1] * 100 
        
        # Combine n-gram and ML confidence with weighted average
        # Give more weight to n-grams if they're found, otherwise rely on ML
        if suspicious_ngrams:
            # If suspicious n-grams found, weight them heavily
            combined_confidence = (ngram_confidence * 0.7) + (ml_confidence * 0.3)
            prediction = combined_confidence > 50
        else:
            # If no suspicious n-grams, rely more on ML but be conservative
            combined_confidence = ml_confidence * 0.6  # Reduce confidence when no clear indicators
            prediction = combined_confidence > 70  # Higher threshold
        
        # Additional checks for obviously legitimate messages
        legitimate_indicators = [
            'hi my name is',
            'how are you',
            'good morning',
            'good afternoon',
            'good evening',
            'thank you',
            'thanks',
            'meeting',
            'appointment',
            'birthday',
            'weekend',
            'conference',
            'groceries',
            'doctor',
            'reminder'
        ]
        
        message_lower = message.lower()
        legitimate_count = sum(1 for indicator in legitimate_indicators if indicator in message_lower)
        
        if legitimate_count > 0 and not suspicious_ngrams:
            combined_confidence = max(combined_confidence - (legitimate_count * 15), 5)
            prediction = False
        
        # Ensure confidence is reasonable
        combined_confidence = max(min(combined_confidence, 95), 5)
        
        # Determine risk level based on confidence ranges
        if combined_confidence >= 70:
            risk_level = 'High'
        elif combined_confidence >= 40:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        # Update prediction logic - medium and high risk are considered potential scams
        prediction = combined_confidence >= 40  # Medium and high confidence are considered scams
        
        # Get important features
        if prediction:
            feature_importance = self.get_important_features(processed_text)
        else:
            feature_importance = []
        
        # ML-BASED SCAM TYPE PREDICTION (NEW APPROACH)
        scam_type = None
        print(f"=== ML-BASED SCAM TYPE PREDICTION ===")
        print(f"Prediction: {prediction}")
        print(f"Combined confidence: {combined_confidence}")
        print(f"Suspicious ngrams found: {len(suspicious_ngrams) if suspicious_ngrams else 0}")

        # Only predict scam type for actual scam predictions with reasonable confidence
        if prediction and combined_confidence >= 40:
            scam_type = self.predict_scam_type_ml(message)
            print(f"ML predicted scam type: {scam_type}")
        else:
            print("Not predicting scam type - either not a scam or low confidence")

        print(f"Final scam_type: {scam_type}")
        print(f"=== END ML-BASED SCAM TYPE PREDICTION ===")
        
        # Predict platform using the trained model
        platform = self.predict_platform(message)
        
        return { 
            'risk_level': risk_level, 
            'confidence': round(combined_confidence, 1),
            'processed_data': processed_data,
            'important_features': feature_importance,
            'suspicious_ngrams': suspicious_ngrams,
            'debug_info': debug_info,
            'scam_type': scam_type,  # Now comes from ML model
            'platform': platform,    # Platform prediction from the model
            'ml_confidence': round(ml_confidence, 1),
            'ngram_confidence': round(ngram_confidence, 1)
        }
    
    def predict_scam_type_ml(self, message):
        """
        Use ML model to predict scam type based on training data
        Only returns a scam type if the model is confident enough
        """
        try:
            # Check if scam type model is trained
            if not hasattr(self, 'scam_type_model') or not hasattr(self, 'scam_type_vectorizer'):
                print("Scam type model not trained")
                return None
            
            # Process the text
            processed_text = self.preprocess_text(message)['processed_text']
            
            # Vectorize the text using the scam type vectorizer
            X = self.scam_type_vectorizer.transform([processed_text])
            
            # Get prediction probabilities
            probabilities = self.scam_type_model.predict_proba(X)[0]
            classes = self.scam_type_model.classes_
            
            # Find the highest probability and corresponding class
            max_prob_idx = np.argmax(probabilities)
            max_probability = probabilities[max_prob_idx]
            predicted_class = classes[max_prob_idx]
            
            print(f"ML scam type prediction probabilities:")
            for i, (cls, prob) in enumerate(zip(classes, probabilities)):
                print(f"  {cls}: {prob:.3f}")
            print(f"Best prediction: {predicted_class} with {max_probability:.3f} confidence")
            
            # Only return the prediction if confidence is high enough (>60%)
            # and it's not 'none' or 'legitimate'
            if (max_probability > 0.6 and 
                predicted_class not in ['none', 'legitimate'] and
                predicted_class is not None):
                return predicted_class
            else:
                print(f"ML confidence too low ({max_probability:.3f}) or invalid class ({predicted_class})")
                return None
                
        except Exception as e:
            print(f"Error in ML scam type prediction: {e}")
            return None
    
    def predict_platform(self, message):
        """
        Predict the platform based on the training data
        """
        # First try using the trained model if available
        if hasattr(self, 'platform_model') and hasattr(self, 'platform_vectorizer'):
            try:
                # Process the text
                processed_text = self.preprocess_text(message)['processed_text']
                
                # Vectorize the text
                X = self.platform_vectorizer.transform([processed_text])
                
                # Predict platform
                platform = self.platform_model.predict(X)[0]
                
                # Return the predicted platform
                return platform
            except Exception as e:
                print(f"Error predicting platform with model: {e}")
                # Fall back to exact matching
        
        # If model prediction fails or model not available, use exact matching
        # This is a fallback method that looks for exact matches in the training data
        message_lower = message.lower()
        
        # Look for exact matches in the training data
        for _, row in self.train_data.iterrows():
            if message_lower == row['text'].lower():
                return row['platform']
        
        # If no exact match, look for similar messages
        # This is a simple similarity check - in a real system you might use cosine similarity
        for _, row in self.train_data.iterrows():
            if message_lower in row['text'].lower() or row['text'].lower() in message_lower:
                return row['platform']
        
        # If all else fails, return the most common platform in the training data
        platform_counts = self.train_data['platform'].value_counts()
        if not platform_counts.empty:
            return platform_counts.index[0]
        
        # Default fallback
        return "other"
    
    def get_important_features(self, processed_text):
        """
        Get the most important features (words) for the prediction
        """
        # Get feature names from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Transform the processed text
        X = self.vectorizer.transform([processed_text])
        
        # Get the TF-IDF scores for each word
        tfidf_scores = dict(zip(feature_names, X.toarray()[0]))
        
        # Sort by score and get top 10
        important_features = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return important_features
 
    def add_feedback(self, message, is_scam, scam_type, platform): 
        """
        Add user feedback to the training data and retrain the model
        """
        print(f"Adding feedback: is_scam={is_scam}, scam_type={scam_type}, platform={platform}")
        print(f"Message: {message[:50]}...")
        
        feedback_data = pd.DataFrame({ 
            'text': [message], 
            'is_scam': [is_scam], 
            'scam_type': [scam_type], 
            'platform': [platform] 
        }) 
        
        # Add to flagged messages
        feedback_data.to_csv('data/flagged_messages.csv', mode='a', header=False, index=False) 
        print("Added to flagged_messages.csv")
        
        # Add to training data
        self.train_data = pd.concat([self.train_data, feedback_data], ignore_index=True) 
        self.train_data.to_csv('data/training_data.csv', index=False) 
        print("Added to training_data.csv")
        
        # Learn new suspicious n-grams if this is a scam message
        new_ngrams = []
        if is_scam:
            print("Learning n-grams from scam message...")
            new_ngrams = self.learn_ngrams_from_feedback(message, is_scam)
            print(f"Learned {len(new_ngrams)} new n-grams")
        
        # Retrain the model with the new data
        self.train_model() 
        print("Model retrained successfully")
        
        return {
            'status': 'success',
            'message': 'Feedback added and model retrained successfully.',
            'new_training_size': len(self.train_data),
            'new_ngrams': new_ngrams if is_scam and new_ngrams else []
        }
 
    def get_stats(self): 
        train_data = pd.read_csv('data/training_data.csv') 
        flagged_data = pd.read_csv('data/flagged_messages.csv') 
         
        stats = { 
            'total_samples': len(train_data), 
            'scam_samples': len(train_data[train_data['is_scam'] == True]), 
            'non_scam_samples': len(train_data[train_data['is_scam'] == False]), 
            'flagged_messages': len(flagged_data), 
            'flagged_scams': len(flagged_data[flagged_data['is_scam'] == True]), 
            'flagged_non_scams': len(flagged_data[flagged_data['is_scam'] == False]), 
            'scam_types': train_data['scam_type'].value_counts().to_dict(), 
            'platforms': train_data['platform'].value_counts().to_dict(),
            'suspicious_ngrams': {
                'bigrams': len(self.suspicious_bigrams),
                'trigrams': len(self.suspicious_trigrams)
            }
        } 
         
        return stats

    def learn_ngrams_from_feedback(self, message, is_scam):
        """
        Learn new suspicious n-grams from user feedback with strict filtering
        Only learn n-grams that are actually suspicious, not common phrases
        """
        if not is_scam:
            return []  # Only learn from scam messages
        
        # Extract bigrams and trigrams from the message
        text_lower = message.lower()
        bigrams = self.extract_ngrams(text_lower, 2)
        trigrams = self.extract_ngrams(text_lower, 3)
        
        # Define common innocent words/phrases that should NOT be learned as suspicious
        innocent_words = {
            'here', 'am', 'doing', 'sale', 'for', 'the', 'and', 'you', 'me', 'my', 'your',
            'have', 'has', 'will', 'can', 'could', 'would', 'should', 'this', 'that',
            'with', 'from', 'they', 'them', 'their', 'our', 'we', 'us', 'him', 'her',
            'his', 'she', 'he', 'it', 'its', 'was', 'were', 'been', 'being', 'are',
            'is', 'am', 'do', 'does', 'did', 'get', 'got', 'go', 'going', 'come',
            'coming', 'see', 'saw', 'look', 'looking', 'know', 'knew', 'think',
            'thought', 'say', 'said', 'tell', 'told', 'ask', 'asked', 'give', 'gave',
            'take', 'took', 'make', 'made', 'work', 'working', 'time', 'day', 'week',
            'month', 'year', 'today', 'tomorrow', 'yesterday', 'now', 'then', 'when',
            'where', 'what', 'who', 'why', 'how', 'which', 'some', 'any', 'all',
            'many', 'much', 'more', 'most', 'less', 'few', 'little', 'big', 'small',
            'good', 'bad', 'new', 'old', 'first', 'last', 'next', 'other', 'same',
            'different', 'right', 'left', 'up', 'down', 'in', 'out', 'on', 'off',
            'over', 'under', 'above', 'below', 'between', 'through', 'during',
            'before', 'after', 'while', 'since', 'until', 'because', 'if', 'unless',
            'although', 'though', 'however', 'therefore', 'so', 'but', 'or', 'nor',
            'either', 'neither', 'both', 'not', 'no', 'yes', 'maybe', 'perhaps',
            'probably', 'certainly', 'definitely', 'really', 'very', 'quite', 'rather',
            'too', 'also', 'even', 'still', 'already', 'yet', 'just', 'only', 'almost',
            'nearly', 'about', 'around', 'approximately'
        }
        
        # Define suspicious keywords that should be part of learned n-grams
        suspicious_keywords = {
            'click', 'verify', 'confirm', 'urgent', 'immediate', 'account', 'suspended',
            'locked', 'security', 'alert', 'warning', 'compromised', 'unauthorized',
            'password', 'login', 'pin', 'cvv', 'ssn', 'social', 'bank', 'credit',
            'card', 'payment', 'transfer', 'money', 'cash', 'bitcoin', 'crypto',
            'lottery', 'winner', 'prize', 'claim', 'congratulations', 'selected',
            'limited', 'expires', 'deadline', 'final', 'notice', 'action', 'required',
            'suspend', 'terminate', 'block', 'freeze', 'update', 'details', 'information',
            'personal', 'confidential', 'secret', 'private', 'send', 'provide',
            'enter', 'submit', 'complete', 'process', 'validate', 'authenticate'
        }
        
        # Get existing suspicious n-grams in the message
        suspicious_result = self.get_suspicious_ngrams(message)
        existing_suspicious = suspicious_result['ngrams']
        existing_ngram_texts = [item['ngram'] for item in existing_suspicious]
        
        new_suspicious = []
        
        # Function to check if an n-gram is worth learning
        def is_worth_learning(ngram_text):
            words = ngram_text.split()
            
            # Don't learn if all words are innocent
            if all(word in innocent_words for word in words):
                return False
            
            # Don't learn if it doesn't contain any suspicious keywords
            if not any(word in suspicious_keywords for word in words):
                return False
            
            # Don't learn very short or common phrases
            if len(ngram_text) < 6:  # Very short phrases
                return False
            
            # Don't learn if it's already known
            if ngram_text in existing_ngram_texts:
                return False
            
            return True
        
        # Learn selective bigrams
        for bigram in bigrams:
            if (bigram not in self.suspicious_bigrams and 
                is_worth_learning(bigram) and 
                len(bigram.split()) == 2):
            
                self.suspicious_bigrams[bigram] = 0.7
                new_suspicious.append({
                    'ngram': bigram,
                    'score': 0.7,
                    'type': 'bigram'
                })

        # Learn selective trigrams
        for trigram in trigrams:
            if (trigram not in self.suspicious_trigrams and 
                is_worth_learning(trigram) and 
                len(trigram.split()) == 3):
            
                self.suspicious_trigrams[trigram] = 0.7
                new_suspicious.append({
                    'ngram': trigram,
                    'score': 0.7,
                    'type': 'trigram'
                })

        # Save the updated suspicious n-grams only if we learned something meaningful
        if new_suspicious:
            self.save_suspicious_ngrams()
            print(f"Learned {len(new_suspicious)} new meaningful suspicious patterns")
        else:
            print("No new meaningful suspicious patterns found to learn")

        return new_suspicious
