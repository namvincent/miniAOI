import subprocess
import asyncio
from flask_cors import CORS
from flask import Flask, render_template
from menu import menu,take_sample,TakeCoordinates
from async_checking import async_checking

app = Flask(__name__)
CORS(app)

@app.route('/')
async def home():
    # result = await async_checking()
    return 'Hello World'

@app.route('/test')
def test():
    try:
        # Replace 'your_command_here' with the actual command you want to run
        result = subprocess.check_output(['python3','main.py','0'], shell=True, stderr=subprocess.STDOUT)
        return f"Command executed successfully: {result.decode('utf-8')}"
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.output.decode('utf-8')}"

@app.route('/takeSample')
async def takeSample():
    await take_sample()
    await TakeCoordinates()
    return 'Finish!'

@app.route('/visualInspection')
async def visualInspection():
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # result = loop.run_until_complete(checking())
    
    result = await async_checking()

    # future = executor.submit(checking())
    # while not future.done():
    #     asyncio.sleep(0.1)
    return result

async def checking():
    await async_checking()
    return 'done'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
