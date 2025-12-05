#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_SAMPLES 1000
#define GRID_SIZE 50
#define LEARNING_RATE 0.1
#define MAX_EPOCHS 1000

typedef struct {
    double x1;
    double x2;
    int target;
} DataPoint;

typedef struct {
    double weights[2];
    double bias;
    int trained;
} Perceptron;

// Funciones de archivos
int leer_datos(const char* filename, DataPoint datos[], int* max_samples);
void guardar_visualizacion(const char* filename, const char* visualizacion);

// Funciones del perceptrón
void inicializar_perceptron(Perceptron *p);
int predecir(Perceptron *p, double x1, double x2);
void entrenar_perceptron(Perceptron *p, DataPoint datos[], int num_datos);
double calcular_precision(Perceptron *p, DataPoint datos[], int num_datos);

// Funciones de visualización
void normalizar_datos(DataPoint datos[], int num_datos, double *min_x1, double *max_x1, double *min_x2, double *max_x2);
char* generar_visualizacion_ascii(Perceptron *p, DataPoint datos[], int num_datos);

// Utilidades
void limpiar_pantalla();
void mostrar_progreso(int epoch, int error);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <archivo_datos> <archivo_salida>\n", argv[0]);
        printf("Ejemplo: %s formatoDatos.txt grafiASSII.txt\n", argv[0]);
        return 1;
    }

    limpiar_pantalla();
    printf("PERCEPTRON CON VISUALIZACION ASCII\n");
    printf("==================================\n");
    printf("Archivo de datos: %s\n", argv[1]);
    printf("Archivo de salida: %s\n", argv[2]);
    printf("\n");

    // Leer datos
    DataPoint datos[MAX_SAMPLES];
    int num_datos = 0;
    
    if (!leer_datos(argv[1], datos, &num_datos)) {
        printf("Error leyendo archivo de datos\n");
        return 1;
    }
    
    printf("Datos cargados: %d muestras\n", num_datos);

    // Entrenar perceptrón
    Perceptron p;
    inicializar_perceptron(&p);
    
    printf("\nEntrenando perceptron...\n");
    entrenar_perceptron(&p, datos, num_datos);
    
    double precision = calcular_precision(&p, datos, num_datos);
    printf("Precision: %.2f%%\n", precision * 100);

    // Generar y guardar visualización
    printf("\nGenerando visualizacion ASCII...\n");
    char* visualizacion = generar_visualizacion_ascii(&p, datos, num_datos);
    guardar_visualizacion(argv[2], visualizacion);
    
    printf("Visualizacion guardada en: %s\n", argv[2]);
    
    free(visualizacion);
    return 0;
}

int leer_datos(const char* filename, DataPoint datos[], int* num_datos) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        return 0;
    }
    
    char line[256];
    *num_datos = 0;
    
    while (fgets(line, sizeof(line), file) && *num_datos < MAX_SAMPLES) {
        // Saltar comentarios y líneas vacías
        if (line[0] == '#' || line[0] == '\n') continue;
        
        // Leer x1, x2, target
        if (sscanf(line, "%lf %lf %d", 
                  &datos[*num_datos].x1, 
                  &datos[*num_datos].x2, 
                  &datos[*num_datos].target) == 3) {
            (*num_datos)++;
        }
    }
    
    fclose(file);
    return *num_datos > 0;
}

void guardar_visualizacion(const char* filename, const char* visualizacion) {
    FILE *file = fopen(filename, "w");
    if (file) {
        fprintf(file, "%s", visualizacion);
        fclose(file);
    }
}

void inicializar_perceptron(Perceptron *p) {
    srand(time(NULL));
    p->weights[0] = ((double)rand() / RAND_MAX) * 2 - 1;
    p->weights[1] = ((double)rand() / RAND_MAX) * 2 - 1;
    p->bias = ((double)rand() / RAND_MAX) * 2 - 1;
    p->trained = 0;
}

int predecir(Perceptron *p, double x1, double x2) {
    double suma = p->weights[0] * x1 + p->weights[1] * x2 + p->bias;
    return suma >= 0 ? 1 : 0;
}

void entrenar_perceptron(Perceptron *p, DataPoint datos[], int num_datos) {
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        int error_total = 0;
        
        for (int i = 0; i < num_datos; i++) {
            int prediccion = predecir(p, datos[i].x1, datos[i].x2);
            int error = datos[i].target - prediccion;
            error_total += abs(error);
            
            // Actualizar pesos
            p->weights[0] += LEARNING_RATE * error * datos[i].x1;
            p->weights[1] += LEARNING_RATE * error * datos[i].x2;
            p->bias += LEARNING_RATE * error;
        }
        
        if (epoch % 100 == 0) {
            mostrar_progreso(epoch, error_total);
        }
        
        if (error_total == 0) {
            printf("Convergencia en epoca %d\n", epoch);
            break;
        }
    }
    p->trained = 1;
}

double calcular_precision(Perceptron *p, DataPoint datos[], int num_datos) {
    int correctos = 0;
    for (int i = 0; i < num_datos; i++) {
        if (predecir(p, datos[i].x1, datos[i].x2) == datos[i].target) {
            correctos++;
        }
    }
    return (double)correctos / num_datos;
}

void normalizar_datos(DataPoint datos[], int num_datos, double *min_x1, double *max_x1, double *min_x2, double *max_x2) {
    *min_x1 = *max_x1 = datos[0].x1;
    *min_x2 = *max_x2 = datos[0].x2;
    
    for (int i = 1; i < num_datos; i++) {
        if (datos[i].x1 < *min_x1) *min_x1 = datos[i].x1;
        if (datos[i].x1 > *max_x1) *max_x1 = datos[i].x1;
        if (datos[i].x2 < *min_x2) *min_x2 = datos[i].x2;
        if (datos[i].x2 > *max_x2) *max_x2 = datos[i].x2;
    }
}

char* generar_visualizacion_ascii(Perceptron *p, DataPoint datos[], int num_datos) {
    double min_x1, max_x1, min_x2, max_x2;
    normalizar_datos(datos, num_datos, &min_x1, &max_x1, &min_x2, &max_x2);
    
    // Crear buffer para la visualización (más grande para seguridad)
    char* buffer = malloc(GRID_SIZE * (GRID_SIZE + 1) * 20);
    int pos = 0;
    
    // Cabecera (solo ASCII estándar)
    pos += sprintf(buffer + pos, "VISUALIZACION PERCEPTRON ASCII\n");
    pos += sprintf(buffer + pos, "===============================\n\n");
    pos += sprintf(buffer + pos, "Leyenda:\n");
    pos += sprintf(buffer + pos, "  'X' = Clase 1 (Correcto)\n");
    pos += sprintf(buffer + pos, "  'O' = Clase 0 (Correcto)\n");
    pos += sprintf(buffer + pos, "  '!' = Error de clasificacion\n");
    pos += sprintf(buffer + pos, "  '.' = Frontera de decision\n");
    pos += sprintf(buffer + pos, "  ' ' = Espacio de clase 1\n\n");
    
    // Generar grid
    for (int y = GRID_SIZE - 1; y >= 0; y--) {
        pos += sprintf(buffer + pos, "%2d ", y);
        
        for (int x = 0; x < GRID_SIZE; x++) {
            // Convertir coordenadas grid a coordenadas datos
            double x_real = min_x1 + (max_x1 - min_x1) * x / (GRID_SIZE - 1);
            double y_real = min_x2 + (max_x2 - min_x2) * y / (GRID_SIZE - 1);
            
            int prediccion = predecir(p, x_real, y_real);
            
            // Verificar si hay un punto de datos en esta posición
            char simbolo = ' ';
            int es_punto_dato = 0;
            int es_error = 0;
            
            for (int i = 0; i < num_datos; i++) {
                int grid_x = (int)((datos[i].x1 - min_x1) / (max_x1 - min_x1) * (GRID_SIZE - 1));
                int grid_y = (int)((datos[i].x2 - min_x2) / (max_x2 - min_x2) * (GRID_SIZE - 1));
                
                if (grid_x == x && grid_y == y) {
                    es_punto_dato = 1;
                    if (prediccion != datos[i].target) {
                        es_error = 1;
                    }
                    break;
                }
            }
            
            // Determinar símbolo (solo caracteres ASCII estándar)
            if (es_punto_dato) {
                if (es_error) {
                    simbolo = '!';
                } else {
                    // Buscar la clase real del punto
                    for (int i = 0; i < num_datos; i++) {
                        int grid_x = (int)((datos[i].x1 - min_x1) / (max_x1 - min_x1) * (GRID_SIZE - 1));
                        int grid_y = (int)((datos[i].x2 - min_x2) / (max_x2 - min_x2) * (GRID_SIZE - 1));
                        
                        if (grid_x == x && grid_y == y) {
                            simbolo = datos[i].target == 1 ? 'X' : 'O';
                            break;
                        }
                    }
                }
            } else {
                // Mostrar frontera de decisión (usamos '.' en lugar de '·')
                double decision = p->weights[0] * x_real + p->weights[1] * y_real + p->bias;
                if (fabs(decision) < 0.1 * (fabs(max_x1 - min_x1) + fabs(max_x2 - min_x2))) {
                    simbolo = '.';
                } else if (prediccion == 0) {
                    simbolo = ' ';
                }
            }
            
            pos += sprintf(buffer + pos, "%c", simbolo);
        }
        pos += sprintf(buffer + pos, "\n");
    }
    
    // Eje X
    pos += sprintf(buffer + pos, "   ");
    for (int x = 0; x < GRID_SIZE; x += 5) {
        pos += sprintf(buffer + pos, "%-5d", x);
    }
    pos += sprintf(buffer + pos, "\n\n");
    
    // Información del modelo
    pos += sprintf(buffer + pos, "Modelo: %.3f*x1 + %.3f*x2 + %.3f = 0\n", 
                  p->weights[0], p->weights[1], p->bias);
    pos += sprintf(buffer + pos, "Rango datos: x1[%.2f, %.2f] x2[%.2f, %.2f]\n",
                  min_x1, max_x1, min_x2, max_x2);
    
    return buffer;
}

void limpiar_pantalla() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

void mostrar_progreso(int epoch, int error) {
    printf("Epoca %d - Error: %d\n", epoch, error);
}
