import tkinter as tk
from tkinter import messagebox

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.geometry("400x600")
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.lista_colunas = [] 
        self.entradas = []
        colunas = ["listado1", "listado2", "listado3", "listado4", "listado5", "nome_lista"]
        
        #adicionando logo a aplicacao
        logo = tk.PhotoImage(file='app\prod\logo-banco-safra-256.png')
        logo_label = tk.Label(self, image=logo)
        logo_label.pack(side='top')
        for i in range(5):
            self.lista_colunas.append(tk.Label(self, text=colunas[i]))
            self.lista_colunas[i].pack(side="top")

            self.entradas.append(tk.Entry(self))
            self.entradas[i].pack(side="top")

        self.variable = tk.StringVar(self)
        self.variable.set("Selecione uma lista")
        self.lista_colunas.append(tk.Label(self, text=colunas[5]))
        self.lista_colunas[5].pack(side="top")
        self.selecao_lista = tk.OptionMenu(self, self.variable, "lista1", "lista2", "lista3")
        self.selecao_lista.pack(side="top")

        self.confirmar_listado1 = tk.Button(self, text="Enviar", command=self.submit_input)
        self.confirmar_listado1.pack(side="bottom")

    def submit_input(self):
        valores = []
        for i in range(5):
            valores.append(self.entradas[i].get())
            if not valores[i]:
                tk.messagebox.showerror("Erro",f"O campo {self.lista_colunas[i]['text']} deve ser preenchido!")
                return
            print(f"Você digitou no campo {self.lista_colunas[i]['text']}: {valores[i]}")

        lista_selecionada = self.variable.get()
        if lista_selecionada == "Selecione uma lista":
            tk.messagebox.showerror("Erro","Para prosseguir, favor selecionar uma lista")
            return
        print(f"Você selecionou a lista: {lista_selecionada}")

root = tk.Tk()
app = Application(master=root)
app.mainloop()
