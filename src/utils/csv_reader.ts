import { rejects } from 'assert'
import fs from 'fs'
import path from 'path'

export class CsvReader {


    // Método para importar o csv e retornar um objeto 
    // com as amostras e com as saídas desejadas
    public static async importCsv(dir: string): Promise<number[][]> {

        try {

            // Ler o arquivo csv
            const csv: string = await this.read(dir)

            // Transforma o csv lido em um array de valores
            return this.csvToArrayObj(csv)

        } catch (error) {
            throw error
        }


    }

    // Método para ler o arquivo csv
    private static read(dir: string): Promise<string> {
        return new Promise((resolve, reject) => {
            fs.readFile(dir, 'utf8', (error, data) => {
                if (error)
                    reject(error)
                else
                    resolve(data)
            })
        })
    }

    // Transofrma o csv lido em um array de objetos
    private static csvToArrayObj(csv: string): number[][] {

        // Separa cada linha do csv em uma entrada
        const inputs: string[] = csv.split('\r\n')

        // Cria variável da amostra
        let samples: number[][] = []

        inputs.map(input => {

            const objs: number[] = []

            input.split(',').map(value => {
                objs.push(parseFloat(value))
            })

            samples.push(objs)

        })

        return samples
    }


}