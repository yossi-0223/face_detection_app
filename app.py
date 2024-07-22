from flask import Flask, render_template, Response
from yunet_camera import Video

app=Flask(__name__)

# /にアクセスするとindex.htmlを返す
@app.route('/')
def index():
    return render_template('index.html')

# 表示する画像を含むHTTP responseを逐次出力するジェネレータ
def gen(camera):
    while True:
        frame=camera.get_frame_overlay()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

# /videoにアクセスするとResponse(gen(Video))を返す
@app.route('/video')
def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)