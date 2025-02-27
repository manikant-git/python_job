import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

class TicketClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = OneVsRestClassifier(LinearSVC())
        self.department_mapping = {
            0: 'Technical Support',
            1: 'Billing',
            2: 'Account Management',
            3: 'Product Inquiry',
            4: 'Bug Report'
        }
        self.priority_mapping = {
            0: 'Low',
            1: 'Medium',
            2: 'High',
            3: 'Urgent'
        }

    def preprocess_ticket(self, text):
        return str(text).lower().strip()

    def train(self, tickets_data):
        X = [self.preprocess_ticket(ticket) for ticket in tickets_data['description']]
        X = self.vectorizer.fit_transform(X)
        y_dept = tickets_data['department']
        self.classifier.fit(X, y_dept)
        
        self.priority_keywords = {
            'urgent': 3, 'emergency': 3, 'critical': 3, 'down': 3,
            'error': 2, 'failed': 2, 'broken': 2,
            'issue': 1, 'problem': 1,
            'question': 0, 'how to': 0
        }

    def classify_ticket(self, ticket_text):
        processed_text = self.preprocess_ticket(ticket_text)
        vectorized_text = self.vectorizer.transform([processed_text])
        
        department_id = self.classifier.predict(vectorized_text)[0]
        department = self.department_mapping[department_id]
        
        priority_score = 0
        for keyword, score in self.priority_keywords.items():
            if keyword in processed_text:
                priority_score = max(priority_score, score)
        
        priority = self.priority_mapping[priority_score]
        
        return {
            'ticket_text': ticket_text,
            'department': department,
            'priority': priority,
            'auto_response': self.generate_auto_response(department, priority)
        }

    def generate_auto_response(self, department, priority):
        responses = {
            'Technical Support': 'Our technical team will investigate your issue.',
            'Billing': 'Our billing department will review your concern.',
            'Account Management': 'An account manager will assist you shortly.',
            'Product Inquiry': 'Our product team will provide information.',
            'Bug Report': 'Our development team will analyze this bug report.'
        }
        
        eta = "2 hours" if priority == "Urgent" else "24 hours" if priority == "High" else "48 hours"
        return f"{responses[department]} Expected response time: {eta}"
