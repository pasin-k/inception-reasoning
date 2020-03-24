from tkinter import *
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image, ImageTk


class LimeGUI:
    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.label = Label(master, text="LIME Interactive")
        self.label.pack()

        self.canvas = Canvas(master, width=300, height=300)
        self.canvas.pack()

        self.greet_button = Button(master, text="Check", command=self.fetch_image)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def fetch_image(self):
        img = image.load_img("./data/test/car_01.jpg", target_size=(299, 299))
        x = image.img_to_array(img).astype(np.uint8)

        self.img = ImageTk.PhotoImage(image=Image.fromarray(x))
        self.canvas.create_image(20, 20, anchor="nw", image=self.img)


root = Tk()
my_gui = LimeGUI(root)
root.mainloop()
