# Anthropic-Finance-Agent

This repository contains the code for a finance-focused chatbot that utilizes the Anthropic Claude model to respond to finance-related questions. The bot identifies financial topics within user queries and provides contextually relevant answers, along with follow-up questions to guide deeper exploration. The chatbot covers key finance topics like inflation, stock markets, interest rates, and general economic trends.

## Technical Overview

The chatbot is built using Python and features an interactive Gradio interface for user interaction. By categorizing finance-related questions, the bot tailors responses from the Claude model to match the user's query, creating a more relevant conversational experience.

### Key Components

1. **Question Classification**:
   - The `identify_question_type` function detects keywords in user queries to categorize them into specific finance topics, such as "inflation," "stock market," or "interest rates."
   - This classification helps the bot contextualize responses and provide relevant follow-up questions to enhance user engagement.

2. **Anthropic Claude Integration**:
   - The `talk_to_anthropic` function interfaces with the Anthropic Claude model via an API key stored in a `.env` file.
   - The function sends the user's prompt to Claude (using model version `"claude-3-opus-20240229"`) and retrieves a response with a maximum of 300 tokens.

3. **Dynamic Follow-Up Questions**:
   - Based on the `question_type`, the `get_follow_up_question` function dynamically generates follow-up questions that prompt users to explore related aspects of the original query.

4. **Gradio Interface**:
   - A Gradio `Blocks` interface provides the chatbot’s interactive user experience, with `Chatbot`, `Textbox`, and `Button` components for smooth interaction and chat history reset options.

### Functions Overview

- **identify_question_type(prompt)**: Classifies the query into a finance-related category to enable targeted responses.
- **get_follow_up_question(question_type)**: Provides a follow-up question relevant to the query type.
- **talk_to_anthropic(prompt)**: Connects to the Claude model to retrieve AI-generated responses.
- **respond_to_user(message, history)**: Manages the conversation flow by processing inputs, appending responses, and adding follow-ups.
- **clear_chat()**: Resets the chat history to start a new conversation.

### Environment Setup

Set up the Anthropic API key in a `.env` file:
```plaintext
ANTHROPIC_API_KEY=your_api_key_here
```

### Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/finance-assistant-chatbot.git
cd finance-assistant-chatbot
pip install -r requirements.txt
```

### Running the Application

Start the Gradio interface:
```bash
python anthropic_finance_agent.py
```
