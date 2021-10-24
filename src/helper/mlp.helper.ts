import { FeedForAwardOutPut } from "../interfaces/feed_for_award_output"
import { Layer } from "../models/layer"

// Cria a rede neural com base na estrutura enviada
export function createNeuralNetwork(structure: number[]): Layer[] {

    const layers: Layer[] = []

    for (let l = 1; l < structure.length; l++) {
        const layer = new Layer(structure[l], structure[l - 1] + 1)
        layers.push(layer)
    }

    return layers


}

// Calcula a lista de potenciais de ativação (I)
export function getI(inputs: number[], layer: Layer): number[] {

    let arrI: number[] = []

    for (let j = 0; j < layer.weights.length; j++) {

        // Potencial de ativação
        let u: number = 0

        for (let i = 0; i < layer.weights[j].length; i++) {

            u = u + layer.weights[j][i] * inputs[i]

        }

        arrI.push(u)

    }

    return arrI

}

// Calcula a lista de saídas da camada (Y)
export function getY(arrI: number[]) {

    const arrY: number[] = []

    arrI.map(value => {

        arrY.push(Math.tanh(value))

    })

    return arrY

}

export function feedForAward(inputs: number[], layers: Layer[]): FeedForAwardOutPut {

    // Matriz de potenciais de ativação
    const matI: number[][] = []

    // Matriz de saídas das camadas
    const matY: number[][] = []

    for (let l = 0; l < layers.length; l++) {

        // Recebe a lista de potenciais de ativação calculado na camada atual
        let arrI: number[] = []


        if (l == 0) {

            // Se for a primeira camada, usa as entradas das amostras (acrescentado com -1 do bias) 
            // como entradas da camada atual
            arrI = getI([-1, ...inputs], layers[l])

        } else {

            // A partir da segunda camada, usa as saídas da camada anterior (acrescentado com -1 do bias)
            // como entradas da camada atual
            arrI = getI([-1, ...matY[l - 1]], layers[l])

        }

        // Adiciona a lista de potenciais de ativação À matriz
        matI.push(arrI)
        // Calcula a lista de saídas da camada atual e adiciona-o à matriz
        matY.push(getY(arrI))

    }

    return { matI: matI, matY: matY }

}

// Trazer o erro quadrático
export function squareError(expectedOutputs: number[], outputs: number[]) {

    let eq: number = 0

    for (let j = 0; j < expectedOutputs.length; j++) {
        eq = eq + Math.pow((expectedOutputs[j] - outputs[j]), 2)
    }

    return eq / 2

}

// Função para trazer o erro quadrático médio
export function eqm(layers: Layer[], samples: number[][], expectedOutputs: number[][]): number {

    let eqm: number = 0
    const p: number = samples.length

    for (let k = 0; k < p; k++) {
        const outputs: number[] = feedForAward(samples[k], layers).matY[layers.length - 1]
        eqm = eqm + squareError(expectedOutputs[k], outputs)
    }

    return eqm / p

}

// Derivada da função tangente hiperbólica
export function tanhDerivated(value: number): number {
    var e = Math.E
    var negValue = value * (-1)
    return 4 / (((e ** negValue) + (e ** value)) ** 2);
}

// Trazer a lista de gradientes da última camada
// Obs: arrY = lista de saídas calculadas pelas última cada, arrI = lista de potenciais de ativação da última camada
export function lastLayerGradients(expectedOutputs: number[], arrY: number[], arrI: number[]): number[] {

    const gradients: number[] = []

    for (let j = 0; j < expectedOutputs.length; j++) {
        gradients.push((expectedOutputs[j] - arrY[j]) * tanhDerivated(arrI[j]))
    }

    return gradients

}

// Trazer a lista de gradientes das camadas escondidas
// Obs: posteriorLayer = camada posterior, posteriorGradients = lista de gradientes da camada posterior
// arrI = lista de potenciais de ativação da camada atual
export function hiddenLayerGradients(posteriorLayer: Layer, posteriorGradients: number[], arrI: number[]): number[] {

    const gradients: number[] = []

    for (let j = 0; j < arrI.length; j++) {

        let sum: number = 0
        for (let k = 0; k < posteriorGradients.length; k++) {
            // OBS: j + 1 que é para "pular" o peso do bias
            sum = sum + (posteriorGradients[k] * posteriorLayer.weights[k][j + 1])
        }

        gradients.push(-1 * sum * tanhDerivated(arrI[j]))

    }

    return gradients

}