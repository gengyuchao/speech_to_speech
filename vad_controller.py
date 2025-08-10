# vad_controller.py
import threading
import queue

class VadController:
    def __init__(self):
        self._sensitivity = 0.6
        self._is_playing = False
        self._play_sensitivity_factor = 0.2
        self._lock = threading.RLock()  # 使用可重入锁
        self._command_queue = queue.Queue()  # 命令队列
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        """后台工作线程处理命令"""
        while self._running:
            try:
                command, value = self._command_queue.get(timeout=0.1)
                if command == 'set_playing':
                    self._set_playing_internal(value)
                elif command == 'set_sensitivity':
                    self._set_sensitivity_internal(value)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"VAD 控制器工作线程错误: {e}")

    def _set_playing_internal(self, playing: bool):
        """内部设置播放状态（线程安全）"""
        with self._lock:
            old_state = self._is_playing
            self._is_playing = playing
            if old_state != playing:
                status = "播放中" if playing else "播放结束"
                # print(f"[VAD状态] {status} - 当前阈值: {self.get_threshold():.3f}")

    def _set_sensitivity_internal(self, value: float):
        """内部设置敏感度（线程安全）"""
        with self._lock:
            self._sensitivity = max(0.0, min(1.0, value))
            print(f"基础VAD敏感度设置为: {self._sensitivity}")

    def set_playing(self, playing: bool):
        """异步设置播放状态"""
        try:
            self._command_queue.put(('set_playing', playing), block=False)
        except queue.Full:
            print("VAD 控制命令队列已满")

    def set_sensitivity(self, value: float):
        """异步设置敏感度"""
        try:
            self._command_queue.put(('set_sensitivity', value), block=False)
        except queue.Full:
            print("VAD 控制命令队列已满")

    def get_threshold(self):
        with self._lock:
            if self._is_playing:
                return min(0.95, self._sensitivity + self._play_sensitivity_factor)
            return self._sensitivity

    def stop(self):
        """停止控制器"""
        self._running = False