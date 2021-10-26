import path from "path"
import { CsvReader } from "./utils/csv_reader"
import { Mlp } from "./mlp"


// Taxa de aprendizagem (n)
const learningRate: number = 0.05

// Taxa de precisão requerida (e)
const precision: number = 0.0000001

// Estrutura da rede neural
// 
const structure: number[] = [2, 2, 1]


CsvReader.importCsv(path.join(__dirname, '../csv/samples.csv'))
    .then(samples => {

        CsvReader.importCsv(path.join(__dirname, '../csv/outputs.csv'))
            .then(expectedOutputs => {

                const mlp = new Mlp(samples, expectedOutputs, learningRate, precision, structure)

                mlp.learn()

                console.log(mlp.prediction([
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]
                ]))

            })

    })
