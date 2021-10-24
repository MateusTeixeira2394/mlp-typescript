import { createNeuralNetwork, eqm, feedForAward } from "./helper/mlp.helper"
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
            
            const response: FeedForAwardOutPut = feedForAward(this.samples[k],this.layers) 

            // Passo 6.2.1: Trazer a lista de I de cada camada
            const matI: number[][] = response.matI

            // Passo 6.2.2: Trazer a lista de Y de cada camada
            const matY: number[][] = response.matY

            // Passo 6.2.3: Trazer a lista de gradientes de cada camada

         }

      }


   }

   private calculateEqm(): number {



      return 0
   }


   public getLayers(): readonly Layer[] {
      return this.layers
   }

   public getEpoch(): number {
      return this.epoch
   }

}