document.addEventListener("DOMContentLoaded", function() {
  const chatLog = document.getElementById('chat-log');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');

  sendBtn.addEventListener('click', function() {
    sendMessage();
  });

  userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
      sendMessage();
    }
  });

  function sendMessage() {
    const userMessage = userInput.value.trim();
    if (userMessage !== '') {
      appendMessage('You', userMessage);
      userInput.value = '';
      chatLog.scrollTop = chatLog.scrollHeight;

      // Make AJAX request to Flask backend
      fetch('/chatbot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({message: userMessage})
      })
      .then(response => response.json())
      .then(data => {
        const botResponse = data.response;
        appendMessage('Chatbot', botResponse);
        chatLog.scrollTop = chatLog.scrollHeight;
      })
      .catch(error => {
        console.error('Error:', error);
      });
    }
  }

  function appendMessage(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatLog.appendChild(messageDiv);
  }
});
