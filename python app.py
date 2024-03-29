from flask import Flask, request, jsonify

app = Flask(Force.d_2-Chatbot)

# Define responses for different types of queries
responses = {
    "admission": "For admission queries, please contact the admission office.",
    "courses": "Our college offers a variety of courses. You can find more information on our website.",
    "fees": "Information regarding fees can be found on our college website or by contacting the finance department.",
    "contact": "You can reach us at contact@example.com or call us at +123456789."
}

# Define a function to analyze user queries and generate responses
def analyze_query(query):
    query = query.lower()
    response = "I'm sorry, I don't understand that query."
    
    # elect appropriate response
    if 'admission' in query:
        response = responses["admission"]
    elif 'courses' in query:
        response = responses["courses"]
    elif 'fees' in query:
        response = responses["fees"]
    elif 'contact' in query:
        response = responses["contact"]
    
    return response

@app.route('script.js', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data['message']
    bot_response = analyze_query(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
