import { backPropagation, createNeuralNetwork, eqm, feedForAward, module } from "./helper/mlp.helper"
import { FeedForAwardOutPut } from "./interfaces/feed_for_award_output"
import { Layer } from "./models/layer"

export class Mlp {

   // Obtem um conjunto de amostras (x)
   private samples: number[][]

   // Obtem um conjunto de saídas desejadas associados às amostras (d)
   private expectedOutputs: number[][]

   // Taxa de aprendizagem (n)
   private learningRate: number

   // Taxa de precisão requerida (e)
   private precision: number

   // Número de épocas
   private epoch: number = 0

   // Lista com os potenciais de ativação das camadas
   private arrI: number[][] = []

   // Lista com as saídas das camadas
   private arrY: number[][] = []

   // Camadas da rede neural
   private layers: Layer[] = []

   constructor(samples: number[][], expectedOutputs: number[][], learningRate: number, precision: number, structure: number[]) {
      this.samples = samples
      this.expectedOutputs = expectedOutputs
      this.learningRate = learningRate
      this.precision = precision
      this.layers = createNeuralNetwork(structure)
   }


   public learn() {

      // Passo 1: Obter conjunto de amostras de treinamento

      // Passo 2: Obter conjunto de saídas desejadas

      // Passo 3: Iniciar o cojunto de pesos sinápticos com valores aleatórios

      // Passo 4: Especificar a taxa de aprendizagem e a precisão

      // Passo 5: Iniciar o contator de número de épocas com zero
      this.epoch = 0

      // Passo 6: Repetir as instruções até |eqmBefore - eqmAfter| <= precisão
      let running: boolean = true

      while (running) {

         // Passo 6.1: Obter o erro quadrático médio antes dos ajustes de peso
         const eqmBefore = eqm(this.layers, this.samples, this.expectedOutputs)

         // Passo 6.2: Para cada uma das amostras faça:
         for (let k = 0; k < this.samples.length; k++) {

            const response: FeedForAwardOutPut = feedForAward(this.samples[k], this.layers)

            // Passo 6.2.1: Trazer a lista de I de cada camada
            const matI: number[][] = response.matI

            // Passo 6.2.2: Trazer a lista de Y de cada camada
            const matY: number[][] = response.matY

            // Ajustes dos pesos sinápticos
            backPropagation(this.layers, this.samples[k], this.expectedOutputs[k], this.learningRate, matI, matY)

         }

         // Incrementa o contador de épocas
         this.epoch = this.epoch + 1
         console.log(this.epoch)
         
         const eqmAfter = eqm(this.layers, this.samples, this.expectedOutputs)

         // Se o módulo da diferença entre o Eqm aterior e o Eqm posterior for
         // menor ou igual à precisão é porque o neurônio aprendeu com as amostras
         if (module(eqmAfter - eqmBefore) <= this.precision)
            running = false

      }

   }

   public prediction(samples: number[][]): number[][] {

      const precditions: number[][] = []

      samples.map(inputs => {

         // Traz a resposta do feed for award
         const response: FeedForAwardOutPut = feedForAward(inputs, this.layers)
         
         // A predição da rede neural é a saída da última camada
         precditions.push(response.matY[response.matY.length - 1])

      })

      return precditions

   }

   public getLayers(): readonly Layer[] {
      return this.layers
   }

   public getEpoch(): number {
      return this.epoch
   }

}