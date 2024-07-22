import cv2

# 顔検出モデルの設定
weights = 'face_detection_yunet_2023mar.onnx'
score_threshold = 0.7
faceDetect = cv2.FaceDetectorYN.create(weights, '', (0, 0), score_threshold=score_threshold)
# 画像パス
img_path = 'laughing.png'
# オーバーレイする画像のサイズのパラメタ
fore_size = 1.4

def overlay(frame, fore, faces):
    '''frame内の顔に画像foreをオーバーレイしframeをinplaceに書き換える'''
    # 各画像のサイズを取得
    frame_shape = frame.shape
    fore_og_shape = fore.shape
    for face in faces:
        ratio = fore_size * max(face[2]/fore_og_shape[0], face[3]/fore_og_shape[1])
        fore_resized = cv2.resize(fore, dsize = None, fx = ratio, fy = ratio)
        fore_shape = fore_resized.shape
        # foreを表示する位置（左上の頂点）を計算
        location = [int(face[0] + (1/2) * (face[3] - fore_shape[1])), int(face[1] + (1/2) * (face[2] - fore_shape[0]))]
        # アルファオーバーレイ
        x1, y1 = max(location[0], 0), max(location[1], 0)
        x2 = min(location[0] + fore_shape[1], frame_shape[1])
        y2 = min(location[1] + fore_shape[0], frame_shape[0])
        ax1, ay1 = x1 - location[0], y1 - location[1]
        ax2, ay2 = ax1 + x2 - x1, ay1 + y2 - y1
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - fore_resized[ay1:ay2, ax1:ax2, 3:] / 255) + \
                            fore_resized[ay1:ay2, ax1:ax2, :3] * (fore_resized[ay1:ay2, ax1:ax2, 3:] / 255)

# カメラから画像を取得して顔を囲う枠を上書きするクラス
class Video(object):
    def __init__(self): 
        # デフォルトのカメラを取得．
        self.video=cv2.VideoCapture(0)
        self.fore = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    def __del__(self):
        # カメラを閉じる
        self.video.release()
    def get_frame(self):
        # カメラからフレームを取得．ret (boolean): 取得の成否，frame: 画像を表現するndarray (行数x列数x3色)
        _, frame=self.video.read()
        # 左右反転
        frame = cv2.flip(frame, 1)
        _, frame = cv2.imencode('.jpg', frame)
        # 画像(ndarray)をbyteに変換
        return frame.tobytes()
    def get_frame_overlay(self):
        # カメラからフレームを取得．ret (boolean): 取得の成否，frame: 画像を表現するndarray (行数x列数x3色)
        _, frame=self.video.read()
        # 左右反転
        frame = cv2.flip(frame, 1)
        # フレームのサイズを設定
        height, width = frame.shape[:2]
        faceDetect.setInputSize((width, height))
        # 検出された顔のリストを取得
        _, faces = faceDetect.detect(frame)
        faces = faces if faces is not None else []
        # bounding boxの情報のみ抽出
        # face = (左上のx座標, 左上のy座標, 幅，高さ)
        faces = [list(map(int, face[:4])) for face in faces]
        print('faces:', faces)
        # 重ねる画像をロード
        fore = self.fore
        # 画像をオーバーレイ
        overlay(frame, fore, faces)
        _, frame = cv2.imencode('.jpg', frame)
        # 画像(ndarray)をbyteに変換
        return frame.tobytes()