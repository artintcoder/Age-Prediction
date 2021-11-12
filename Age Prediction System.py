#Artint Coder
#Age Prediction
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import math
import argparse

def main():
    root = Tk()
    app = Window1(root)
    root.mainloop()
    return
 

class Window1:

    def __init__(self, master):
        
        self.master = master
        self.master.title("Age Prediction")
        self.master.geometry('630x600+400+100')
        self.master.config(bg='black')
        self.frame = Frame(self.master)
        self.frame.pack()
        
        
        self.CORONA = StringVar()
        self.Path = StringVar()
        self.Cor = StringVar()

        #=================================== Title =================================================
        self.TitleFrame = Frame(self.master, relief = 'ridge', bd = 4,bg = 'white')
        self.TitleFrame.place(x=0, y=0,width = 630, height = 40)
        
        self.lblTitle = Label(self.TitleFrame,  text = 'Age Prediction',bg='white', fg= 'black', font = ('times new romana',15,'bold'))
        self.lblTitle.place(x=240, y=0)
    
        #================================= Frames ===========================================
        
        self.MainFrame = LabelFrame(self.master,  relief = 'ridge', bd = 6,bg = 'white')
        self.MainFrame.place(x=0, y=45,width = 630, height = 555)
        
        self.loadimg2 = Image.open("img/ages.jpg")
        self.renderimg2 = ImageTk.PhotoImage(self.loadimg2)
        self.img2_Frame = Label(self.MainFrame, image=self.renderimg2, relief = 'ridge', bd = 10,bg = 'black')
        self.img2_Frame.image = self.renderimg2
        self.img2_Frame.place(x=10, y=10 ,width = 600, height=430)
        
        self.BtnMainFrame = Frame(self.MainFrame, relief = 'ridge', bd = 8,bg = 'black')
        self.BtnMainFrame.place(x=10, y=500,width = 600, height = 40)

        # The warning sentence in the second frame
        self.COR_Label = Entry(self.MainFrame, fg='black',  font = ('times new romana',15,'bold'), relief = 'ridge', bd = 4,bg = 'white', textvariable= self.Cor)
        self.COR_Label.place(x=10, y=445,width = 600, height = 50 )
        
        self.Test_path = Entry(self.MainFrame ,textvariable= self.Path).place(x=1000, y=445)

        #================================ Button Frame two =====================
        self.BTN_Open_Img =  ttk.Button(self.BtnMainFrame,  style = 'TButton', text = 'Open Image'  , width = 31,command=self.FunOpenImg).grid(row=0, column=0, padx=0, pady=0)
        self.BTN_Open_Cam_img =  ttk.Button(self.BtnMainFrame, text = 'take image'  , style = 'TButton', width = 31,command=self.open_img).grid(row=0, column=1, padx=2, pady=0)
        self.BTN_Open_Cam =  ttk.Button(self.BtnMainFrame, text = 'Open Camera'  , style = 'TButton', width = 30,command=self.FunVidosAge).grid(row=0, column=2, padx=0, pady=0)


        
    def FunOpenImg (self):
        global my_image
        
        self.OpenImgFrame = LabelFrame(self.img2_Frame)
        self.OpenImgFrame.place(x=0, y=0,width = 580, height = 410)
        
        self.OpenImgFrame.filename = filedialog.askopenfilename (initialdir="/gui/images", title="Select A File", filetypes = (("jpg files", "*.jpg"), ("all files", "*.*")))
        my_image = ImageTk.PhotoImage(Image.open(self.OpenImgFrame.filename))
        
        self.Path.set(self.OpenImgFrame.filename)

        self.TwoFramee = Label(self.OpenImgFrame,  image = my_image )
        self.TwoFramee.place(x=0, y=0,width = 577, height = 428)
        self.FunImagAge()

      #===================================================================================                                                         
    def FunTakeImg (self):
        videoCaptureObject = cv2.VideoCapture(0)
        result = True
        while(result):
            ret,frame = videoCaptureObject.read()
            cv2.imwrite("img/NewPicture.jpg",frame)
            result = False
        videoCaptureObject.release()
        cv2.destroyAllWindows()

    def open_img(self):
        self.OpenCamFrame = LabelFrame(self.img2_Frame)
        self.OpenCamFrame.place(x=0, y=0,width = 580, height = 410)
        
        self.FunTakeImg()
        load = Image.open("img/NewPicture.jpg")
        render = ImageTk.PhotoImage(load)
   
        self.img1_Frame = Label(self.OpenCamFrame, image=render)
        self.img1_Frame.image = render
        self.Path.set("img/NewPicture.jpg")
        self.img1_Frame.place(x=0, y=0 ,width = 577, height=428)
        self.FunImagAge()
        #=================================================================================
        #function to open image     
    def FunImagAge(self):

        def highlightFace(net, frame, conf_threshold=0.7):
            frameOpencvDnn=frame.copy()
            frameHeight=frameOpencvDnn.shape[0]
            frameWidth=frameOpencvDnn.shape[1]
            blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        
            net.setInput(blob)
            detections=net.forward()
            faceBoxes=[]
            for i in range(detections.shape[2]):
                confidence=detections[0,0,i,2]
                if confidence>conf_threshold:
                    x1=int(detections[0,0,i,3]*frameWidth)
                    y1=int(detections[0,0,i,4]*frameHeight)
                    x2=int(detections[0,0,i,5]*frameWidth)
                    y2=int(detections[0,0,i,6]*frameHeight)
                    faceBoxes.append([x1,y1,x2,y2])
                    cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
            return frameOpencvDnn,faceBoxes
        
        
        parser=argparse.ArgumentParser()
        parser.add_argument('--image')
        
        args=parser.parse_args()
        
        faceProto="opencv_face_detector.pbtxt"
        faceModel="opencv_face_detector_uint8.pb"
        ageProto="age_deploy.prototxt"
        ageModel="age_net.caffemodel"
        
        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        faceNet=cv2.dnn.readNet(faceModel,faceProto)
        ageNet=cv2.dnn.readNet(ageModel,ageProto)
        
        video=cv2.VideoCapture(args.image if args.image else 0)
        padding=20
        
        frame= cv2.imread(self.Path.get())

        
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        
        if not faceBoxes:
            self.Cor.set("No face detected")
        
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                       min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                       :min(faceBox[2]+padding, frame.shape[1]-1)]
        
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            COR = (f'Hello.. your old is between ({age[1:-1]}) years')
            self.Cor.set(COR)
           
      #function to open camera  
    def FunVidosAge (self):

        def highlightFace(net, frame, conf_threshold=0.7):
            frameOpencvDnn=frame.copy()
            frameHeight=frameOpencvDnn.shape[0]
            frameWidth=frameOpencvDnn.shape[1]
            blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        
            net.setInput(blob)
            detections=net.forward()
            faceBoxes=[]
            for i in range(detections.shape[2]):
                confidence=detections[0,0,i,2]
                if confidence>conf_threshold:
                    x1=int(detections[0,0,i,3]*frameWidth)
                    y1=int(detections[0,0,i,4]*frameHeight)
                    x2=int(detections[0,0,i,5]*frameWidth)
                    y2=int(detections[0,0,i,6]*frameHeight)
                    faceBoxes.append([x1,y1,x2,y2])
                    cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
            return frameOpencvDnn,faceBoxes
        
        
        parser=argparse.ArgumentParser()
        parser.add_argument('--image')
        
        args=parser.parse_args()
        
        faceProto="opencv_face_detector.pbtxt"
        faceModel="opencv_face_detector_uint8.pb"
        ageProto="age_deploy.prototxt"
        ageModel="age_net.caffemodel"
        
        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        faceNet=cv2.dnn.readNet(faceModel,faceProto)
        ageNet=cv2.dnn.readNet(ageModel,ageProto)
        
        video=cv2.VideoCapture(args.image if args.image else 0)
        padding=20
        while cv2.waitKey(1)<0:
            hasFrame,frame=video.read()
            if not hasFrame:
                cv2.waitKey()
                break
        
            resultImg,faceBoxes=highlightFace(faceNet,frame)

            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                           min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                           :min(faceBox[2]+padding, frame.shape[1]-1)]
        
                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
          
                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]

                cv2.putText(resultImg, f' {age} years', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Detecting age ", resultImg)

        
if __name__ == '__main__':
    main()