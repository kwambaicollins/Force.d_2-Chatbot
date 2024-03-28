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
      // Here you can add logic to handle user input and generate bot response
      // For simplicity, I'll just echo back the user message
      setTimeout(function() {
        appendMessage('Chatbot', userMessage);
        userInput.value = '';
        chatLog.scrollTop = chatLog.scrollHeight;
      }, 500);
    }
  }

  function appendMessage(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatLog.appendChild(messageDiv);
  }
});
