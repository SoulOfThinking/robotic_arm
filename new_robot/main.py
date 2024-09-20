import urx
import time
import pyaudio
import wave
import json
import random
import requests
import threading
import subprocess
import parameters as para
import visual_util as visual
import asyncio
import matplotlib.pyplot as plt
from datetime import datetime
from gripper import Gripper
from shapely.geometry import Point, Polygon
import utils as tools

# 配置音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav"

PI = 3.14159
SQRT2 = 1.41421

pos_history = ["2"] # 机械臂初始位置
stack_num = 0 # 堆叠数

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Command failed with exit code {result.returncode}: {result.stderr}")
    return result.stdout

def nt():
    return "["+datetime.now().strftime("%H:%M:%S.%f")+"] "

def record():
    audio = pyaudio.PyAudio()

    # 打开音频流
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    # 标志变量，用于控制录音的开始和结束
    recording = False

    def listen_for_input():
        nonlocal recording
        input("按下回车键开始录音...")
        recording = True
        print("录音中... 按下回车键结束录音")
        input()  # 等待用户第二次按下回车
        recording = False
        print("录音结束...")

    # 创建并启动监听用户输入的线程
    input_thread = threading.Thread(target=listen_for_input)
    input_thread.start()

    frames = []

    while input_thread.is_alive():
        if recording:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError as e:
                print(f"Error recording: {e}")
                break

    # 停止录音
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 保存录音文件
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"已保存到 {WAVE_OUTPUT_FILENAME}")

def get_instructions(audio_text):
    url = 'http://10.50.0.37:5001/chat'
    input_data = [
        {'role': 'system', 'content': para.PROMPT},
        {'role': 'user', 'content': audio_text}
    ]

    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(input_data))
    if response.status_code == 200:
        response_data = response.json()
        return response_data
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
    
def get_visual_info(text, image_path="capture_fixed.jpg", history=None):
    url = 'http://10.50.0.37:5000/process'
    data = {'text': text}
    
    if history is not None:
        data['history'] = str(history)
    
    if image_path is None:
        response = requests.post(url, data=data)
    else:
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        response_data = response.json()
        # print("Response:", response_data)
    else:
        print("Failed to get response. Status code:", response.status_code)
        response_data = None

    return response_data

def get_object_position(obj_name):
    response = get_visual_info(f"框出图中{obj_name}的位置", "capture_fixed.jpg")
    # 返回文本的格式类似于：'<ref>裁纸刀</ref><box>(258,259),(622,985)</box>'
    xml_text = response['response']
    if "<box>" not in xml_text:
        return None
    else:
        print(xml_text)
    box_start = xml_text.find("<box>") + len("<box>")
    box_end = xml_text.find("</box>")
    box_coords = xml_text[box_start:box_end]
    
    try:
        p1, p2 = eval(box_coords)
    except:
        print(box_coords)
        return None

    h, w = visual.image_size()
    x = (p1[0] + p2[0]) / 2.0 * (w / 1000)
    y = (p1[1] + p2[1]) / 2.0 * (h / 1000)

    return x, y

def is_pos_valid(pos):
    point = Point(pos[:2])
    table = Polygon(para.TABLE_CORNER.values())
    return table.contains(point)

def pipeline_visual(rob, gripper):
    async def grab():
        await gripper.move(255, 255, 255)
    async def release():
        await gripper.move(0, 255, 255)
    
    record()# 记录音频文件，文件名为output.wav
    command = tools.stt(WAVE_OUTPUT_FILENAME)
    print(command)
   #  command = f"hear -d -i {WAVE_OUTPUT_FILENAME} -l zh-CN"

    try:
        # audio_text = run_command(command)
        audio_text = command
    except Exception:
        print("run_command")
        raise KeyboardInterrupt
    print(nt()+"识别语音：", audio_text.replace("\n", ""))

    audio_text = audio_text.replace("材质刀", "裁纸刀").replace("彩纸刀", "裁纸刀").replace("才知道", "裁纸刀")

    response_data = get_instructions(audio_text)
    instruction = json.loads(response_data['message']['content'].replace("json\n", "").replace("`", ""))

    print(nt()+"解析指令：", instruction)
    obj = instruction['obj']
    action = instruction['action']
    dir, targ = instruction['pos'] if instruction['pos'] != "" else ["", ""]
    msg = instruction['msg']

    # return

    if action == "move_to": # 移动物体
        visual.cam_capture_frame()
        visual.cam_calibration()

        obj_pixel = get_object_position(obj)
        obj_pos = visual.pixel_to_table(obj_pixel)

        if type(targ) == int and targ in [1, 2]:
            targ_pos = para.UR_TPOSE[str(targ)]
        elif type(targ) == str:
            if "1" in targ:
                targ_pos = para.UR_TPOSE["1"]
            elif "2" in targ:
                targ_pos = para.UR_TPOSE["2"]
            else:
                targ_pixel = get_object_position(targ)
                targ_pos = visual.pixel_to_table(targ_pixel)
        else:
            return
        # return

        # 检查目标位置是否可达
        if not is_pos_valid(targ_pos):
            print(nt()+"目标位置不可到达:", targ_pos)
            # subprocess.Popen(f"say -v Lilian 抱歉，目标位置不可到达", shell=True)
            tools.tts("抱歉，目标位置不可到达!","target_not_get.wav")
            tools.play_wav_file("target_not_get.wav")
            return
        if not is_pos_valid(obj_pos):
            print(nt()+"物体位置不可到达:", obj_pos)
            # subprocess.Popen(f"say -v Lilian 抱歉，物体位置不可到达", shell=True)
            tools.tts("抱歉，物体位置不可到达!","object_not_get.wav")
            tools.play_wav_file("object_not_get.wav")
            return

        # subprocess.Popen(f"say -v Lilian {msg}", shell=True) # 语音回复
        tools.tts(msg,"msg.wav")
        tools.play_wav_file("msg.wav")

        if dir == "at": # 放在位置上
            cord = (targ_pos[0], targ_pos[1])
            z_up = 0.1
            z_down = 0.005
            orientation = (0, PI, 0)
        elif dir == "near": # 放在旁边（随机）
            offset = random.choice([
                (0.1, random.uniform(-0.1, 0.1)), # 左边
                (-0.1, random.uniform(-0.1, 0.1)), # 右边
                (random.uniform(-0.1, 0.1), -0.1), # 前面
                (random.uniform(-0.1, 0.1), 0.1) # 后面
            ])
            cord = (targ_pos[0] + offset[0], targ_pos[1] + offset[1])
            z_up = 0.1
            z_down = 0.005
            orientation = (0, PI, 0) if abs(offset[0]) < abs(offset[1]) else (PI/SQRT2, PI/SQRT2, 0)
        elif dir == "stack": # 堆叠
            global stack_num
            temp = input(f"请输入目标位置已有堆叠数（默认为{stack_num+1}）：")
            stack_num = int(temp) if temp != "" else stack_num + 1
            cord = (targ_pos[0], targ_pos[1])
            z_up = 0.1 + stack_num * 0.05
            z_down = 0.005 + stack_num * 0.05
            orientation = (0, PI, 0)

        rob.movel((obj_pos[0], obj_pos[1], 0.1, 0, PI, 0), acc=1, vel=1) # 移动到物体上方
        rob.movel((obj_pos[0], obj_pos[1], 0.005, 0, PI, 0), acc=0.5, vel=0.1) # 夹爪下移

        loop = asyncio.get_event_loop()
        loop.run_until_complete(grab()) # 抓取
        time.sleep(1)

        rob.movel((obj_pos[0], obj_pos[1], 0.1, 0, PI, 0), acc=3, vel=1) # 夹爪上移
        rob.movel(cord+(z_up,)+orientation, acc=1, vel=0.7) # 移动到目标上方
        rob.movel(cord+(z_down,)+orientation, acc=1, vel=0.5) # 夹爪下移

        loop.run_until_complete(release()) # 释放
        time.sleep(0.5)

        rob.movel(cord+(z_up,)+orientation, acc=3, vel=1) # 夹爪上移
        rob.movel(para.UR_TPOSE["start"], acc=3, vel=1) # 移动到初始位置
    elif action == "move": # 移动机械臂
        if dir not in ["front", "back", "left", "right", "up", "down"]:
            print(nt()+"目标位置格式错误")
            # subprocess.Popen(f"say -v Lilian 抱歉，没有听懂要移动到哪里", shell=True)
            tools.tts("抱歉，没有听懂要移动到哪里","move_not_hear.wav")
            tools.play_wav_file("move_not_hear.wav")
            return
        if type(targ) == int:
            targ_pos = targ/100.0
        else:
            targ_pos = 0.1
        now_pos = rob.getl()
        if dir == "left":
            target = (now_pos[0] + targ_pos, now_pos[1], now_pos[2], now_pos[3], now_pos[4], now_pos[5])
        elif dir == "right":
            target = (now_pos[0] - targ_pos, now_pos[1], now_pos[2], now_pos[3], now_pos[4], now_pos[5])
        elif dir == "front":
            target = (now_pos[0], now_pos[1] - targ_pos, now_pos[2], now_pos[3], now_pos[4], now_pos[5])
        elif dir == "back":
            target = (now_pos[0], now_pos[1] + targ_pos, now_pos[2], now_pos[3], now_pos[4], now_pos[5])
        elif dir == "up":
            target = (now_pos[0], now_pos[1], now_pos[2] + targ_pos, now_pos[3], now_pos[4], now_pos[5])
        elif dir == "down":
            target = (now_pos[0], now_pos[1], now_pos[2] - targ_pos, now_pos[3], now_pos[4], now_pos[5])
        # else:
        #     print(nt()+"移动方向错误")
        #     subprocess.Popen(f"say -v Lilian 抱歉，没有明白移动方向", shell=True)
        #     return

        if not is_pos_valid(target):
            print(nt()+"目标位置不可到达:", target)
       #     subprocess.Popen(f"say -v Lilian 抱歉，目标位置不可到达", shell=True)
            tools.tools.tts("抱歉，目标位置不可到达","target_not_catch.wav")
            tools.play_wav_file("target_not_catch.wav")            
            return

      #  subprocess.Popen(f"say -v Lilian {msg}", shell=True) # 语音回复
        tools.tts(msg,"msg.wav")
        tools.play_wav_file("msg.wav")         
        rob.movel(target, acc=1, vel=0.5) # 移动到目标位置
    elif action == "grab":
        visual.cam_capture_frame()
        visual.cam_calibration()

        pixel = get_object_position(obj)
        print(pixel)
        pos = visual.pixel_to_table(pixel)
        print(pos)

        if not is_pos_valid(pos):
            print(nt()+"物体位置不可到达:", pos)
        #    subprocess.Popen(f"say -v Lilian 抱歉，物体位置不可到达", shell=True)
            tools.tts("抱歉，物体位置不可到达","object_not_catch.wav")
            tools.play_wav_file("object_not_catch.wav") 
            return

      #  subprocess.Popen(f"say -v Lilian {msg}", shell=True) # 语音回复
        tools.tts(msg,"msg.wav")
        tools.play_wav_file("msg.wav") 

        rob.movel((pos[0], pos[1], 0.1, 0, PI, 0), acc=1, vel=1) # 移动到物体上方
        rob.movel((pos[0], pos[1], 0.005, 0, PI, 0), acc=1, vel=0.1) # 夹爪下移

        loop = asyncio.get_event_loop()
        loop.run_until_complete(grab()) # 抓取
        time.sleep(1)

        rob.movel((pos[0], pos[1], 0.1, 0, PI, 0), acc=3, vel=1) # 夹爪上移
    elif action == "put":
        # subprocess.Popen(f"say -v Lilian {msg}", shell=True)
        tools.tts(msg,"msg.wav")
        tools.play_wav_file("msg.wav") 
        pos = rob.getl()
        rob.movel((pos[0], pos[1], 0.005, 0, PI, 0), acc=1, vel=1)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(release())
        time.sleep(1)

        rob.movel((pos[0], pos[1], 0.1, 0, PI, 0), acc=3, vel=1)
        rob.movel(para.UR_TPOSE["start"], acc=3, vel=1)
    return

def test(rob, gripper):
    visual.cam_capture_frame()
    visual.cam_calibration()

    # obj = "黄色钳子"
    # pixel = get_object_position(obj)

    pixel = visual.annotation(num=1)
    plt.close()
    print(pixel)
    pos = visual.pixel_to_table(pixel[0])
    print(pos)
    # pos[1] -= 0.01

    # exit()
    rob.movel((pos[0], pos[1], 0.23, 0, PI, 0), acc=1, vel=1)
    rob.movel((pos[0], pos[1], 0.13, 0, PI, 0), acc=1, vel=0.1)

    async def grab():
        await gripper.move(255, 255, 255)
    # asyncio.run(grab())
    loop = asyncio.get_event_loop()

    # 在当前事件循环中运行异步任务
    loop.run_until_complete(grab())
    time.sleep(1)
    # exit()

    rob.movel((pos[0], pos[1], 0.23, 0, PI, 0), acc=1, vel=1)
    targ = para.UR_TPOSE["target"]
    rob.movel(targ, acc=1, vel=0.5)
    rob.movel((targ[0],targ[1], 0.14, 0, PI, 0), acc=1, vel=0.5)


    # exit()

    async def release():
        await gripper.move(0, 255, 255)
    # asyncio.run(release())
    loop.run_until_complete(release())
    time.sleep(1)
    rob.movel((targ[0],targ[1], 0.23, 0, PI, 0), acc=1, vel=0.5)
    rob.movel(para.UR_TPOSE["start"], acc=1, vel=1)
    return

def plot_test():
    visual.cam_capture_frame()
    visual.cam_calibration()

    pixel = visual.annotation(num=1)
    plt.close()
    print(pixel)
    pos = visual.pixel_to_table(pixel[0])
    print(pos)

def test_mapping(rob, draw_table=False):

    test_x = [0.6, 0.4, 0.2, 0, -0.2, -0.4]
    test_y = [-0.6, -0.4, -0.2]

    if draw_table:
        for y in test_y:
            for x in test_x:
                rob.movel((x, y, 0.23, 0, PI, 0), acc=1, vel=1)
                rob.movel((x, y, 0.1915, 0, PI, 0), acc=1, vel=1)
                rob.movel((x, y, 0.23, 0, PI, 0), acc=1, vel=1)
                time.sleep(1)
    
    visual.cam_capture_frame()
    visual.cam_calibration()
    visual.annotation(num=len(test_x)*len(test_y))

def init(full=False, annotation=False):
    # 机械臂
    while True:
        try:
            rob = urx.Robot(para.UR_IP)
            break
        except:
            print(nt()+"连接机械臂失败，正在重试...", end="")
            time.sleep(0.5)
    rob.set_tcp(para.UR_TCP)
    rob.set_payload(para.UR_PAYLOAD, para.UR_PAYLOAD_CENTER)
    print(nt()+f"已连接到机械臂：{para.UR_IP}, 位于{rob.getl()}")
    #  print("连接到机械臂")
    
    rob.movel(para.UR_TPOSE["start"], acc=0.5, vel=1)

    print(nt()+"移动到初始位置")
    # rob.close()

    # 夹爪
    gripper = Gripper(para.UR_IP)
    async def gripper_init():
        await gripper.connect()
        # await gripper.activate()
    # asyncio.run(gripper_init())
    loop = asyncio.get_event_loop()

    # # 在当前事件循环中运行异步任务
    loop.run_until_complete(gripper_init())
    print(nt()+"夹爪已连接")

    if full:
        # LLM模块
        get_instructions("回到2号位置")
        print(nt()+"LLM已连接")

        # VLM模块
        get_visual_info("这是什么")
        print(nt()+"VLM已连接")

    # 视觉模块
    visual.cam_capture_frame() # 拍摄一张capture.jpg
    # visual.cal_calibration_matrix() # 计算两个摄像头校准矩阵
    visual.cam_calibration() # 校准得到capture_fixed.jpg
    if annotation:
        visual.cal_transform_matrix(plot=True) # 计算变换矩阵
    
    return rob, gripper

if __name__ == "__main__":
    try:
        rob, gripper = init(annotation=False)

        while True:
            pipeline_visual(rob, gripper)
            # test(rob, gripper)
        # plot_test()
        # test_mapping(rob=None, draw_table=False)
    except KeyboardInterrupt:
        rob.close()
        print(nt()+"机械臂连接已关闭，程序退出")
        exit()
    # pipeline_visual()