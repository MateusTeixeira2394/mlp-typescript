export class Neuron {

    // Pesos
    private weights: number[] = []

    // Bias ou Limiar
    private bias: number = 0

    // Inicia a classe informando a quantidade de entradas
    // O neurônio é iniciado com pesos e o seu bias (limiar) aleatórios
    constructor(entriesQuantities: number) {

        this.generateWeights(entriesQuantities);
        this.bias = this.getRandomNumber()

    }

    // Método para gerar os pesos aleatoriamente
    private generateWeights(entriesQuantities: number) {

        // Para cada quatidade de entrada, será gerado um novo peso aleatório
        for (let i = 0; i < entriesQuantities; i++) {

            this.weights.push(this.getRandomNumber())

        }

    }

    // Método para trazer um valor aleatório entre 0 e 1
    private getRandomNumber(): number {
        // Retorna um valor aleatório de até 2 casas decimais
        return Number(Math.random().toFixed(2))
    }

    // Método para trazer o potêncial de ativação (u)
    public getActivationPotential(inputs: number[]): number {

        let sum = 0

        // Para cada valor de entrada, multiplica pelo seu respectivo peso
        // e acumula o resultado
        inputs.map((value, i) => {

            sum = sum + value * this.weights[i]

        })

        // O potencial de ativação é o somatório do produto dos pesos com os
        // valores de entrada menos o valor do bias (limiar)
        return sum - this.bias

    }

    // Método para atualizar os pesos
    public adjustWeights(learningRate: number, gradient: number, previousOutputs: number[]): void {

        // Ajusta todos os pesos do neurônio
        // i = posição do peso
        this.weights.map((weight, i) => {

            weight = weight + (learningRate * gradient * previousOutputs[i])

        })

        // Ajusta o Bias (Liminar)
        // Note que a entrada do Bias (Limiar) é sempre -1
        this.updateBias(learningRate, gradient, -1)

    }

    // Método para atualziar o Bias (Limiar) 
    // taxa de aprendizagem = n, saída desejada = d, potencial de ativação = u, entradas da amostra atual = x[]
    private updateBias(learningRate: number, gradient: number, previousOutput: number): void {
        this.bias = this.bias + (learningRate * gradient * previousOutput)
    }

    public getWeights() {
        return this.weights
    }

    public getBias() {
        return this.bias
    }
}