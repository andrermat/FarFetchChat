class History():
    def __init__(self, prompt, limit=20):
        self.limit = limit
        self.initialPrompt = prompt
        self.history = []

    def add(self, content, role):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.limit:
            self.history.pop(0)

    def get(self):
        res = []
        res.append({"role": "system", "content": self.initialPrompt})
        res.extend(self.history)
        return res



def init_GPT_History():
    chat_prompt = '''
    You will behave as a chat assistant for a clothing website,
    You help customers find products they like and answer their questions,
    The webside will look for products and you will generate responses,
    Your name is iFetch bot
    You will ocasionally receive product descriptions given by the system, you do not need to describe them completely, keep in mind the user can see pictures of the procuts along with your reply so make short descriptions or skip them completely if it is apropriate.
    The following text is the conversation flow between the user and the assistant. You will take the role of the assistant.
    
    '''
    return History(chat_prompt, limit = 10)