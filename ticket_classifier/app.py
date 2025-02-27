from flask import Flask, request, render_template_string
from classifier import TicketClassifier
import pandas as pd

app = Flask(__name__)

# Initialize and train the classifier
classifier = TicketClassifier()

# Sample training data
sample_data = {
    'description': [
        'Cannot login to my account',
        'Need to update billing information',
        'How do I use feature X?',
        'System is down, urgent help needed',
        'Want to upgrade my subscription',
        'Bug in the checkout process',
        'Need to reset password',
        'Billing charge incorrect',
        'Feature request for dashboard',
        'Account locked out'
    ],
    'department': [0, 1, 3, 0, 1, 4, 0, 1, 3, 0]  # Corresponding departments
}

classifier.train(pd.DataFrame(sample_data))

# HTML template with better styling
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Support Ticket Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #34495e;
            font-weight: bold;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .result h2 {
            color: #2c3e50;
            margin-top: 0;
        }
        .result p {
            margin: 10px 0;
        }
        .priority {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
        }
        .Urgent { background-color: #ff4444; color: white; }
        .High { background-color: #ffbb33; color: black; }
        .Medium { background-color: #00C851; color: white; }
        .Low { background-color: #33b5e5; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Support Ticket Classifier</h1>
        <form method="POST">
            <div class="form-group">
                <label>Enter Ticket Description:</label>
                <textarea name="ticket_text" required placeholder="Type your support ticket description here..."></textarea>
            </div>
            <button type="submit">Classify Ticket</button>
        </form>
        {% if result %}
        <div class="result">
            <h2>Classification Result</h2>
            <p><strong>Department:</strong> {{ result.department }}</p>
            <p><strong>Priority:</strong> <span class="priority {{ result.priority }}">{{ result.priority }}</span></p>
            <p><strong>Auto Response:</strong> {{ result.auto_response }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        ticket_text = request.form['ticket_text']
        result = classifier.classify_ticket(ticket_text)
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True)
