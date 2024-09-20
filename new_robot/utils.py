import pyaudio
import wave


def play_wav_file(file_path):
    # 打开 .wav 文件
    wf = wave.open(file_path, 'rb')

    # 创建 PyAudio 流
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 读取并播放音频数据
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    # 关闭流和 PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

import requests
import json
from pydub import AudioSegment

def convert_to_16k(input_wav_path, output_wav_path):
    # 加载音频文件
    audio = AudioSegment.from_wav(input_wav_path)
    
    # 设置采样率为16kHz
    audio = audio.set_frame_rate(16000)
    
    # 保存新的音频文件
    audio.export(output_wav_path, format="wav")

def stt(input_path): 
    # 音频文件的本地路径

    # print('down')
    # Flask 服务端的 URL 和端口号
    url = 'http://10.24.7.245:5000/asr'
    convert_to_16k(input_path,"output.wav")
    # 打开音频文件并读取内容
    with open(input_path, 'rb') as audio_file:
        # 构造文件上传的表单数据
        files = {'file': (audio_file.name, audio_file, 'audio/wav')}
        
        # 发送 POST 请求到 Flask 服务端
        response = requests.post(url, files=files)
        
        # 检查请求是否成功
        if response.status_code == 200:
            # 解析返回的 JSON 数据
            data = response.json()
            # 打印 ASR 结果
            print("ASR Result:", data['result'])
            return(data['result'])
        else:
            # 打印错误信息
            print("Error:", response.json()['error'])

# 输入的是一段字符串
def tts(text, output_name):
    # 定义 Flask 服务端的 URL 和端口
    url = 'http://10.24.7.245:5000/tts'

    # 定义要合成的文本
    # text_to_speak = '今天的天气不错！'

    # 构建请求的参数
    params = {
        'text': text
    }

    # 发送 GET 请求到 Flask 服务端
    response = requests.get(url, params=params)

    # 检查请求是否成功
    if response.status_code == 200:
        # 获取响应的内容
        audio_content = response.content

        # 将语音文件保存到本地
        with open(output_name, 'wb') as audio_file:
            audio_file.write(audio_content)
        print(f'语音文件已保存为{output_name}')
        return(output_name)
    else:
        print('请求失败，状态码：', response.status_code)