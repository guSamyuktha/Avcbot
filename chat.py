import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "AVC"
#print("Let's chat! (type 'quit' to exit)")
def get_response(msg):

#while True:
   # sentence = input("You: ")
    # sentence = "do you use credit cards?"

   # if sentence == "quit":
    #    print("For further details visit avccengg.net")
     #   break

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."
        print('Help me Learn?')
        tag=input('Please enter general category of your question  ')
        flag=-1
        for i in range(len(intents['intents'])):
            if tag.lower() in intents['intents'][i]['tag']:
                intents['intents'][i]['patterns'].append(input('Enter your message: '))
                intents['intents'][i]['responses'].append(input('Enter expected reply: '))        
                flag=1

        if flag==-1:
            intents['intents'].append (
            {'tag':tag,
            'patterns': [input('Please enter your message')],
            'responses': [input('Enter expected reply')]})
        with open('intents.json','w') as outfile:
            outfile.write(json.dumps(intents,indent=4))

if __name__=="__main__":
    print("Lets chat")
    while True:
        sentence=input("you: ")
        if sentence == "quit":
            print("For further details visit avccengg.net")
            break
        res=get_response(sentence)
        print(res)