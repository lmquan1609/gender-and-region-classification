from flask import Flask, render_template, request, redirect
from model import predict, predict_pytorch
from get_audio import get_audio
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

index_label={'female_north':0,'female_central':1,'female_south':2,
            'male_north':3,'male_central':4,'male_south':5}
label_Vn={'female_north':"Nữ - Bắc",'female_central':"Nữ - Trung",         'female_south':'Nữ - Nam',
            'male_north':"Nam - Bắc",'male_central':"Nam - Trung",'male_south':"Nam - Nam"}
        
@app.route("/", methods=["GET", "POST"])
def index():
    predictContent = ""
    path = ""
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        name = request.form['text']
        if name:
            name = get_audio(name)
            y,predictContent = predict_pytorch(name)
            return render_template('index.html', 
                                    predict = label_Vn[predictContent],
                                    female_north = str(y[0][index_label['female_north']]) + " %",
                                    female_central = str(y[0][index_label['female_central']]) + " %",
                                    female_south = str(y[0][index_label['female_south']])+" %",
                                    male_north = str(y[0][index_label['male_north']])+" %",
                                    male_central = str(y[0][index_label['male_central']])+" %",
                                    male_south = str(y[0][index_label['male_south']]) + " %",
                                    path = name)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file= str(file).split(" ")[1].replace("'","")
            path = os.path.join(cur_dir, 'audio', file)
            
            y,predictContent=predict_pytorch(path)
            return render_template('index.html', 
                                    predict = label_Vn[predictContent],
                                    female_north = str(y[0][index_label['female_north']]) + " %",
                                    female_central = str(y[0][index_label['female_central']]) + " %",
                                    female_south = str(y[0][index_label['female_south']])+" %",
                                    male_north = str(y[0][index_label['male_north']])+" %",
                                    male_central = str(y[0][index_label['male_central']])+" %",
                                    male_south = str(y[0][index_label['male_south']]) + " %",
                                    path = path)
    return render_template('index.html', predict=predictContent,path=path)

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=8888)
