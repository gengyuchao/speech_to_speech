import requests

class InferenceHandler:
    def __init__(self, server_url):
        self.server_url = server_url

    def send_inference_request(self,file_path, temperature=0.0, temperature_inc=0.2, response_format='json'):  
        """  
        发送推断请求到本地服务器。  
    
        参数:  
        - file_path: 文件的路径，例如 'output.wav'  
        - temperature: 温度参数，默认为 0.0  
        - temperature_inc: 温度增量参数，默认为 0.2  
        - response_format: 响应格式，默认为 'json'  
    
        返回:  
        - 响应的 JSON 数据（如果 response_format 为 'json'）或原始响应内容  
        """  
        url = 'http://127.0.0.1:8080/inference'  
        files = {'file': open(file_path, 'rb')}  # 使用 'rb' 模式打开文件以二进制形式读取  
        data = {  
            'temperature': str(temperature),  
            'temperature_inc': str(temperature_inc),  
            'response_format': response_format  
        }  
        
        headers = {'Content-Type': 'multipart/form-data'}  
        
        # 注意：在 requests 中，当使用 files 参数时，Content-Type 会被自动设置为 multipart/form-data  
        # 因此，手动设置 headers 中的 Content-Type 可能不是必需的，这里为了清晰说明而保留  
        
        response = requests.post(url, files=files, data=data)  
        
        # 检查响应格式，并返回相应的数据  
        if response.status_code == 200:  
            if response_format == 'json':  
                return response.json()['text']  
            else:  
                return response.text  
        else:  
            return response.status_code, response.text  
    