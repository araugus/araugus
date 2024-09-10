from modelos.restaurante import Restaurante
from modelos.cardapio.bebida import Bebida
from modelos.cardapio.prato import Prato
from modelos.cardapio.sobremesa import Sobremesa

restaurante_japa = Restaurante('DJapa', 'Japonesa')

coca_cola = Bebida('Coca Cola', 5, '500ml')
coca_cola.aplicar_desconto()

sushi = Prato('Sushi', 35, 'Sushi de Salmão')
sushi.aplicar_desconto()

brownie = Sobremesa('Brownie',15,'Fatia')

restaurante_japa.adicionar_no_cardapio(coca_cola)
restaurante_japa.adicionar_no_cardapio(sushi)
restaurante_japa.adicionar_no_cardapio(brownie)

def main():
    restaurante_japa.exibir_cardapio

if __name__ == '__main__':
    main()