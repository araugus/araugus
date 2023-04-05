import tkinter as tk

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
        
        for i in range(6):
            self.lista_colunas.append(tk.Label(self, text=colunas[i]))
            self.lista_colunas[i].pack(side="top")

            self.entradas.append(tk.Entry(self))
            self.entradas[i].pack(side="top")

        self.confirmar_listado1 = tk.Button(self, text="Enviar", command=self.submit_input)
        self.confirmar_listado1.pack(side="bottom")

    def submit_input(self):
        valores = []
        for i in range(6):
            valores.append(self.entradas[i].get())
            print(f"Você digitou no campo {self.lista_colunas[i]['text']}: {valores[i]}")


root = tk.Tk()
app = Application(master=root)
app.mainloop()
