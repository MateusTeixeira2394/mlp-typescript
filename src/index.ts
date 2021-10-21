import path from "path"
import { CsvReader } from "./utils/csv_reader"
import { Mlp } from "./mlp"


// Taxa de aprendizagem (n)
const learningRate: number = 0.005

// Taxa de precisão requerida (e)
const precision: number = 0.00001

const scaffold: number[] = [3, 4, 1]


CsvReader.importCsv(path.join(__dirname, '../csv/samples.csv'))
    .then(samples => {

        CsvReader.importCsv(path.join(__dirname, '../csv/outputs.csv'))
            .then(outputs => {

                
            })

    })