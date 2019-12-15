#импортируем все необходимые библиотеки

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import random

#проверим признаки на корреляцию и исключим признаки, 
#имеющие корреляцию с другими более чем 0.7 по модулю
#и имеющие корреляцию со столбиком стостояние менее 0.1 по модулю

file = pd.read_excel(r'Obschaya_tablitsa.xlsx')
file.corr()

train_data = file[['Номер гильзы', 'Вид', 'Количество заготовок, шт.:',
       'Температура стали, °C:', 'Частота качания, кол-во/мин',
        'Ход кр-ра, мм', 'Скорость разливки, м/мин',
        'Расход воды на кр-р, л/мин', 'Дельта температуры воды, С',
        'Расход воды ЗВО №1, л/мин', 'Расход воды ЗВО №2, л/мин',
        'Расход воды ЗВО №3, л/мин', 'Темп. 2, °С']]

label_data = file[['Состояние']]

#разделим данные на обущающую выборку - 60% и тестовую выборку - 40%

X_train, X_test, y_train, y_test = train_test_split(train_data, label_data, 
                                                    train_size=0.6,
                                                    test_size=0.4, 
                                                    shuffle=True,
                                                   random_state=100)

#переведём данные в тип массива NumPy и заменим отсутствующие значения на 0

X_train_np = np.array(X_train.fillna(0))
X_test_np = np.array(X_test.fillna(0))

y_train_np = np.array(y_train).astype(np.float64)
y_test_np = np.array(y_test).astype(np.float64)

#отстандантизируем данные

scaler = StandardScaler()
scaler.fit(X_train_np, y_train_np)

X = scaler.transform(X_train_np)
X_t = scaler.transform(X_test_np)

#обучим модель k-ближайжих соседей

knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=15, p=1, leaf_size = 15, algorithm='brute')
knn.fit(X, y_train_np)

predictions = knn.predict(X_t)

#вычислим точность нашей модели, вычислив метрику между значениями в таблице
#и предсказаниями классификатора

print('Точность предсказания показателя для детали: {} %'.format((accuracy_score(y_test_np, predictions)*100).round(3)))



from tkinter import *
from tkinter.messagebox import *
from tkinter.filedialog import asksaveasfile, askopenfile
FILE_NAME = NONE

def show_about():
    # второе окно (форма о программе)
    second = Tk()
    second.configure(background='#311e6b')#background='#311e6b'
    second.title('About')
    # this removes the maximize button
    second.resizable(0, 0)
    second.geometry('400x200')
    lb1 = Label(second, text="Прогнозная аналитика для металлургии ПАО 'Кокс'",foreground='#fff', background='#311e6b')
    lb2 = Label(second, text="Точность предсказания показателя для детали: 76.438 %",foreground='#fff',background='#311e6b')
    lb3 = Label(second,foreground='#fff', text="""
    Авторы :    Лаптева Арина Александровна
                Яковлев Павел Андреевич
                       Черепанов Иван Владимирович
    """,background='#311e6b')
    lb4 = Label(second,foreground='#fff', text="2019 год",background='#311e6b')
    bt1 = Button(second,foreground='#fff', text="Закрыть", command=lambda: second.destroy(),background='#311e6b')
    lb1.pack()
    lb2.pack()
    lb3.pack()
    lb4.pack()
    bt1.pack()

def show_help():
    showinfo(title="Help", message="""
                        Predictive-Analitics-for-Metallurgy
Goal: analyze the degree of influence of known factors, select the most significant and build a model of the dependence of run-out of sleeve on these factors
Problems:
●Initial data processing
●Select a model and use training and test samples to train the model
●Analyze the selected model, imagine its advantages and disadvantages
●Write an interface for working in the enterprise
●Write an effective neural network for data analysis
Development stage: data analysis using the method kNN
    """)


root = Tk()
root.configure(background="#4282eb")
root.title("KOKC")
#root.geometry("750x600")
root.iconbitmap(r'PMH.ico')
menuBar = Menu(root)

helpMenu = Menu(menuBar)
helpMenu.configure(background="#4282eb")
menuBar.configure(background="#4282eb")
menuBar.add_cascade(label = "Help", menu  = helpMenu)
helpMenu.add_command(label = "Help", command = show_help)
helpMenu.add_command(label = "About", command = show_about)
menuBar.configure(background="#4282eb")
root.config(menu = menuBar)
root.configure(background="#4282eb")
root.resizable(0,0)
def clear():
    Entry0.delete(0,END)
    Entry1.delete(0,END)
    Entry2.delete(0,END)
    Entry3.delete(0,END)
    Entry4.delete(0,END)
    Entry5.delete(0,END)
    Entry6.delete(0,END)
    Entry7.delete(0,END)
    Entry8.delete(0,END)
    Entry9.delete(0,END)
    Entry10.delete(0,END)
    Entry11.delete(0,END)
    Entry12.delete(0,END)
    lbl = Label(root,bg="#4282eb",foreground="#fff",font='Arial 16', text = "")
    #lbl.grid(row=13, column=0, sticky=W)
    #lbl.pack()
    #lbl.grid_forget()

def inp():
    a = Entry0.get() # берем текст из первого поля
    a = int(a) # преобразуем в число целого типа
    
    b = Entry1.get() 
    b = int(b)
    
    c = Entry2.get() 
    c = int(c)

    d = Entry3.get() 
    d = float(d)

    e = Entry4.get() 
    e = int(e)

    f = Entry5.get() 
    f = int(f)

    g = Entry6.get() 
    g = float(g)

    h = Entry7.get() 
    h = float(h)
    
    i = Entry8.get() 
    i = float(i)

    j = Entry9.get() 
    j = float(j)
    
    k = Entry10.get() 
    k = float(k)
    

    l = Entry11.get() 
    l = float(l)

    m = Entry12.get() 
    m = int(m)
    
    return [a, b, c,d,e,f,g,h,i,j,k,l,m]


def result2(n):
    if n == 0:
        col='#00ff15'
        
        return col
    elif n == 1:
        col='#ffea00'
        
        return col
    elif n == 2:
        col='#ff0000'
        
        return col
def result(n):
    if n == 0:
        print ('У гильзы отсутствует износ')
        a='У гильзы отсутствует износ'
        
        return a
    elif n == 1:
        print('Гильза имеет допустимый износ')
        B = 'Гильза имеет допустимый износ'
        
        return B
    elif n == 2:
        print('Гильза имеет недопустимый износ')
        C = 'Гильза имеет недопустимый износ'
        
        return C

def main():
    data = inp()
    col=result2(int(*knn.predict(scaler.transform([data]))))
    print('col',col)
    result(int(*knn.predict(scaler.transform([data]))))
    q = str(result(int(*knn.predict(scaler.transform([data])))))
    
    lbl = Label(root,bg='#4282eb',foreground=col,font='Arial 16', text = q)
    lbl.grid(row=13, column=0, sticky=W)
    #clear()
    #Entry13.insert(0, result)
    #return answer


# первая метка в строке 0, столбце 0 (0 по умолчанию)
# парамет sticky  означает выравнивание. W, E,N,S — запад, восток, север, юг
Label(root,bg="#4282eb", text='Номер гильзы', font='Arial 16').grid(row=0, sticky=W)

# вторая метка в строке 1
Label(root,bg="#4282eb", text='Вид', font='Arial 16').grid(row=1, sticky=W)
Label(root,bg="#4282eb", text='Количество заготовок, шт.', font='Arial 16').grid(row=2, sticky=W)
Label(root,bg="#4282eb", text='Температура стали, °C', font='Arial 16').grid(row=3, sticky=W)
Label(root,bg="#4282eb", text='Частота качания, кол-во/мин', font='Arial 16').grid(row=4, sticky=W)
Label(root,bg="#4282eb", text='Ход кристаллизатора, мм', font='Arial 16').grid(row=5, sticky=W)
Label(root,bg="#4282eb", text='Скорость разливки, м/мин', font='Arial 16').grid(row=6, sticky=W)
Label(root,bg="#4282eb", text="Расход воды на кристаллизатор, л/мин", font='Arial 16').grid(row=7, sticky=W)
Label(root,bg="#4282eb", text='Дельта температуры воды, °C', font='Arial 16').grid(row=8, sticky=W)
Label(root,bg="#4282eb", text='Расход воды ЗВО №1, л/мин', font='Arial 16').grid(row=9, sticky=W)
Label(root,bg="#4282eb", text='Расход воды ЗВО №2, л/мин', font='Arial 16').grid(row=10, sticky=W)
Label(root,bg="#4282eb", text='Расход воды ЗВО №3, л/мин', font='Arial 16').grid(row=11, sticky=W)
Label(root,bg="#4282eb", text='Температура 2, °С', font='Arial 16').grid(row=12, sticky=W)
#lbl = Label(root,bg="#4282eb",foreground="#fff",font='Arial 20',width =5, text = '')
#lbl.grid(row=13, column=0, sticky=E)

# создаем виджеты текстовых полей
Entry0 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry1 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry2 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry3 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry4 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry5 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry6 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry7 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry8 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry9 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry10 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry11 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
Entry12 = Entry(root, width=10,bg="#c6c8cc", font='Arial 16')
#Entry13 = Entry(root, width=10, font='Arial 16')

# размещаем первые два поля справа от меток, второй столбец (отсчет от нуля)
Entry0.grid(row=0, column=1, sticky=E)
Entry1.grid(row=1, column=1, sticky=E)
Entry2.grid(row=2, column=1, sticky=E)
Entry3.grid(row=3, column=1, sticky=E)
Entry4.grid(row=4, column=1, sticky=E)
Entry5.grid(row=5, column=1, sticky=E)
Entry6.grid(row=6, column=1, sticky=E)
Entry7.grid(row=7, column=1, sticky=E)
Entry8.grid(row=8, column=1, sticky=E)
Entry9.grid(row=9, column=1, sticky=E)
Entry10.grid(row=10, column=1, sticky=E)
Entry11.grid(row=11, column=1, sticky=E)
Entry12.grid(row=12, column=1, sticky=E)
#Entry13.grid(row=13, column=1, sticky=E)


# третье текстовое поле ввода занимает всю ширину строки 2
# columnspan — объединение ячеек по столбцам; rowspan — по строкам
#Entry13.grid(row=13, columnspan=2)

# размещаем кнопку в строке 3 во втором столбце 


but = Button(root, text='Результат', font='Arial 16')
but.grid(row=13, column=1, sticky=E)
but = Button(root, font='Arial 16', text='Результат', command=main)
but.grid(row=13, column=1, sticky=E)
but = Button(root, text='Сброс', font='Arial 16')
but.grid(row=15, column=1, sticky=E)
but = Button(root, font='Arial 16', text='Сброс', command=clear)
but.grid(row=15, column=1, sticky=E)


root.mainloop()