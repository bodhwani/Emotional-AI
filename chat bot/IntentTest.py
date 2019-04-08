from rasa_nlu.model import Interpreter
import json
interpreter = Interpreter.load("./models/current/nlu")
message = "Nissan"
result = interpreter.parse(message)
print(json.dumps(result, indent=2))