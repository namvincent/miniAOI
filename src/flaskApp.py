import subprocess
import asyncio
from flask import Flask, render_template
from main import main,take_sample,TakeCoordinates
from async_checking import async_checkingq

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, Flask on Raspberry Pi!'

@app.route('/test')
def test():
    try:
        # Replace 'your_command_here' with the actual command you want to run
        result = subprocess.check_output(['python3','main.py','0'], shell=True, stderr=subprocess.STDOUT)
        return f"Command executed successfully: {result.decode('utf-8')}"
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.output.decode('utf-8')}"

@app.route('/ff')
def ff():
    take_sample()
    TakeCoordinates()
    return 'ttt'

@app.route('/dd')
def dd():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(checking())
     
    return result

async def checking():
    await async_checkingq()
    return 'done'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
