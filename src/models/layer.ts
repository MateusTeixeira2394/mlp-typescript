export class Layer {

    public weights: number[][] = []

    constructor(neuronsQtd: number, inputsQtd: number) {
        this.generateWeights(neuronsQtd, inputsQtd)
    }

    // Estrutura da rede neural. Cada valor é a quantidade de neurônios na camada
    private generateWeights(neuronsQtd: number, inputsQtd: number) {

        for (let j = 0; j < neuronsQtd; j++) {
            let weight: number[] = []
            for (let i = 0; i < inputsQtd; i++) {
                weight.push(Number(Math.random().toFixed(2)))
            }
            this.weights.push(weight)
        }

    }

}