import numpy as np

class MF():

    def __init__(self, M, K, alpha, beta, iterations):
        """
        Matriz de fatoração para calcular entradas vazias de uma matriz

        Argumentos
        - M (ndarray)   : Matriz
        - K (int)       : Latente
        - alpha (float) : fator de aprendizagem
        - beta (float)  : parametro de regulação
        """

        self.M = M
        self.num_users, self.num_items = M.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Inicializar matriz de características latente de usuário e item
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Inicializar os vetores de tendência
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.M[np.where(self.M != 0)])

        # Criar uma lista para treinamento
        self.samples = [
            (i, j, self.M[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.M[i, j] > 0
        ]

        # Executar método do gradiente para as iterações
        training_process = []
        for i in range(self.iterations):
            #print(i)
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
        return training_process

    def mse(self):
        """
        Calcular erro quadrático médio
        """
        xs, ys = self.M.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.M[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Executar método do gradiente
        """
        for i, j, r in self.samples:
            # Computando a predição e o erro
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Atualizar as têndencias
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Atualizar as matrizes de usuário e itens
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Gerar a predição para o usuário i e item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computar a matriz completa com as predições, P e Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)