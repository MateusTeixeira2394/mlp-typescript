import path from "path"
import { CsvReader } from "./utils/csv_reader"
import { Mlp } from "./mlp"


// Taxa de aprendizagem (n)
const learningRate: number = 0.05

// Taxa de precisão requerida (e)
const precision: number = 0.00000001

// Estrutura da rede neural
// Posição 0: Quantidade de entradas por amostra
// Demais posições: Quantidade de neurônios de cada camada
// Última posição: Além de ser a quantidade de neuronios é a quantidade de saídas
const structure: number[] = [4, 8, 2]


CsvReader.importCsv(path.join(__dirname, '../csv/samples.csv'))
    .then(samples => {

        CsvReader.importCsv(path.join(__dirname, '../csv/outputs.csv'))
            .then(expectedOutputs => {

                const mlp = new Mlp(samples, expectedOutputs, learningRate, precision, structure)

                mlp.learn()

                console.log(mlp.prediction([
                    [4.8, 3.4, 1.6, 0.2],
                    [6.7, 3.1, 4.4, 1.4],
                    [6.0, 2.2, 5.0, 1.5]
                ]))

            })

    })
