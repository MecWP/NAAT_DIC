class ControleRemoto:
    def __init__(self, cor, altura, largura):
        self.cor = cor
        self.altura = altura
        self.largura = largura
        
    def passar_canal(self, botao):
        if botao == '+':
            print('Aumentar o canal')
        elif botao == '-':
            print('Diminuir o canal')
            
controle_remoto = ControleRemoto('preto', '10 cm', '2 cm')
print(controle_remoto.cor)
controle_remoto.passar_canal('+')

controle_remoto2 = ControleRemoto('vermelho', '10 cm', '2 cm')
print(controle_remoto2.cor)
controle_remoto2.passar_canal('-')