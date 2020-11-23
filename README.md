

## ML - Clasificador de perceptrones multicapa

# <div style="text-align: justify">
El clasificador de perceptrones multicapa (MLPC) es un clasificador basado en la red neuronal artificial feedforward . MLPC consta de múltiples capas de nodos. Cada capa está completamente conectada a la siguiente capa de la red. Los nodos de la capa de entrada representan los datos de entrada. Todos los demás nodos mapean entradas a las salidas realizando una combinación lineal de las entradas con los pesos y el sesgo del nodo y aplicando una función de activación. Se puede escribir en forma de matriz para MLPC con capas de la siguiente manera: Los nodos en las capas intermedias usan la función sigmoidea (logística): Los nodos en la capa de salida usan la función softmax: El número de nodos en la capa de salida corresponde al número de clases.wbK+1
 
# <div/>
 
 <img src = "C:\Users\filos\OneDrive\Imágenes\expo1.PNG" >

N

MLPC emplea retropropagación para aprender el modelo. Usamos la función de pérdida logística para la optimización y L-BFGS como rutina de optimización.


## Programa Ejemplo.
```scala
// Se importa MultilayerPerceptronClassifier y MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// $example off$
// Se impor ta la sesion de Spark
import org.apache.spark.sql.SparkSession

/**
 * An example for Multilayer Perceptron Classification.
 */

 // Creacion del objeto MultilayerPerceptronClassifier
object MultilayerPerceptronClassifierExample {

// Se define la funcion main la cual tiene como parametro un Array de tipo string
  def main(args: Array[String]): Unit = {
    // Se crea el objeto de la clase SparkSession, y a la app se le da el nombre de
    // MultilayerPerceptronClassifierExample
    val spark = SparkSession
      .builder
      .appName("MultilayerPerceptronClassifierExample")
      .getOrCreate()

    // $example on$
    // Se cargan los datos en formato libsvm del archivo como un DataFrame
    val data = spark.read.format("libsvm")
      .load("sample_multiclass_classification_data.txt")

    // Se dividen los datos en entrenamiento y prueba
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // Se especifican las capas de la red neuronal:
    // La capa de entrada es de tamaño 4 (caracteristicas), dos capas intermedias
    // una de tamaño 5 y la otra de tamaño 4
    // y 3 de salida (las clases)
    val layers = Array[Int](4, 5, 4, 3)

    // Se establecen los parametros de entrenamiento
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // Se entrena el modelo
    val model = trainer.fit(train)

    // Se calcula la precision de los datos de prueba
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    // Se imprime la exactidud del modelo
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
    // $example off$

    spark.stop()
  }
}
```