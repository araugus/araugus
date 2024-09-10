from modelos.restaurante import Restaurante

restaurante_japa = Restaurante('DJapa', 'Japonesa')
restaurante_japa.receber_avaliacao('Gus', 4)
restaurante_japa.receber_avaliacao('Luis', 5)
restaurante_japa.receber_avaliacao('Amanda', 2)

def main():
    Restaurante.listar_restaurantes()

if __name__ == '__main__':
    main()