#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>

// mpicc grafoDistanciasMPI.c -o grafoDistanciasMPI
// time mpirun -np 4 ./grafoDistanciasMPI -n 2500 -w 20 -d 5 -c 15

/* -------------------- CONSTANTES -------------------- */
#define DEFAULT_NODES 1000
#define DEFAULT_MAX_WEIGHT 10
#define DEFAULT_MIN_EDGES 1
#define DEFAULT_MAX_EDGES 10

#define MAX_NODES 10000
#define MIN_NODES 2
#define MIN_WEIGHT 1
#define MAX_WEIGHT 100000
#define MIN_EDGES_PER_NODE 1
#define MAX_EDGES_PER_NODE 100

/* -------------------- ESTRUCTURAS -------------------- */
// Arista del grafo
typedef struct {
    int to;
    int weight;
} Edge;

// Nodo del grafo con lista de adyacencia
typedef struct {
    Edge edges[MAX_EDGES_PER_NODE];
    int num_edges;
} Node;

// Grafo global
Node graph[MAX_NODES];

/* -------------------- FUNCIONES -------------------- */

// Añade una arista al grafo
void add_edge(int from, int to, int weight) {
    int n = graph[from].num_edges;
    graph[from].edges[n].to = to;
    graph[from].edges[n].weight = weight;
    graph[from].num_edges++;
}

// Genera un grafo aleatorio
void generarGrafo(int nodes, int max_weight, int min_edges, int max_edges) {
    srand(time(NULL));
    for(int i = 0; i < nodes; i++) {
        graph[i].num_edges = 0;
        int edges = rand() % (max_edges - min_edges + 1) + min_edges;
        for(int j = 0; j < edges; j++) {
            int to = rand() % nodes;
            int weight = rand() % max_weight + 1;
            add_edge(i, to, weight);
        }
    }
}

// Devuelve el nodo no visitado con menor distancia
int distanciaMin(int dist[], int visited[], int nodes) {
    int min = INT_MAX, index = -1;
    for(int i = 0; i < nodes; i++) {
        if(!visited[i] && dist[i] < min) {
            min = dist[i];
            index = i;
        }
    }
    return index;
}

// Dijkstra desde un nodo origen hacia todos los demás
void dijkstra(int nodes, int start, int dist[]) {
    int visited[MAX_NODES];
    for(int i = 0; i < nodes; i++) {
        dist[i] = INT_MAX;  // Se inicializan las distancias a infinito
        visited[i] = 0;
    }
    dist[start] = 0;
    for(int i = 0; i < nodes; i++) {
        int u = distanciaMin(dist, visited, nodes);
        if(u == -1) break;
        visited[u] = 1;
        for(int j = 0; j < graph[u].num_edges; j++) {
            int v = graph[u].edges[j].to;
            int weight = graph[u].edges[j].weight;
            if(!visited[v] && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
            }
        }
    }
}

// Calcula estadísticas globales de distancias entre todos los nodos
void calcularEstadisticas(int nodes, int rank, int size) {

    int start = rank * nodes / size;
    int end   = (rank + 1) * nodes / size;

    printf("Proceso %d calculando distancias para nodos [%d, %d)\n", rank, start, end);

    // Estadísticas de distancias mínimas locales
    int local_min_min = INT_MAX;
    int local_max_min= 0;
    long long suma_local_min= 0;
    long long contador_local = 0;

    for(int i = start; i < end; i++) {

        int dist[MAX_NODES];

        dijkstra(nodes, i, dist);

        for(int j = 0; j < nodes; j++) {

            if(i == j) continue;
            if(dist[j] == INT_MAX) continue;

            int val = dist[j];

            if(val < local_min_min) local_min_min = val;
            if(val > local_max_min) local_max_min = val;

            suma_local_min += val;
            contador_local++;
        }
    }

    int global_min_min;
    int global_max_min;
    long long global_sum_min;
    long long global_count;

    MPI_Reduce(&local_min_min, &global_min_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);            //De los valores mínimos calculados en cada proceso (local_min_min), MPI compara todos y selecciona el menor de ellos y se guarda en global_min_min en el proceso 0.
    MPI_Reduce(&local_max_min, &global_max_min, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);            //De los valores máximas calculados en cada proceso (local_max_min), MPI compara todos y selecciona la mayor de ellos y se guarda en global_max_min en el proceso 0.
    MPI_Reduce(&suma_local_min, &global_sum_min, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);     //De las sumas locales calculadas en cada proceso (suma_local_min), MPI suma todas y el resultado se guarda en global_sum_min en el proceso 0.
    MPI_Reduce(&contador_local, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);       //De los contadores locales calculados en cada proceso (contador_local), MPI suma todas y el resultado se guarda en global_count en el proceso 0.

    if(rank == 0){
        printf("\n=== Estadísticas de las distancias MÍNIMAS entre nodos ===\n");
        printf("Mínimo: %d\n", global_min_min);
        printf("Máximo: %d\n", global_max_min);
        printf("Media: %.2f\n", (double)global_sum_min / global_count);
    }

}

/* ======================== MAIN ======================= */
int main(int argc, char *argv[]) {

    int nodes = DEFAULT_NODES;
    int max_weight = DEFAULT_MAX_WEIGHT;
    int min_edges = DEFAULT_MIN_EDGES;
    int max_edges = DEFAULT_MAX_EDGES;

    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "n:w:d:c:")) != -1) {
        switch (c) {
            case 'n': nodes = atoi(optarg); break;
            case 'w': max_weight = atoi(optarg); break;
            case 'd': min_edges = atoi(optarg); break;
            case 'c': max_edges = atoi(optarg); break;
            case '?':
                fprintf(stderr, "Error en los argumentos. Forma de ejecutar el programa: %s [-n nodes] [-w max_weight] [-d min_edges] [-c max_edges]\n", argv[0]);
                return -1;
        }
    }

    /* VALIDACIÓN DE LÍMITES */
    if(nodes < MIN_NODES || nodes > MAX_NODES) {
        fprintf(stderr, "Error: número de nodos (%d) fuera del rango [%d, %d]\n", nodes, MIN_NODES, MAX_NODES);
        return -1;
    }

    if(max_weight < MIN_WEIGHT || max_weight > MAX_WEIGHT) {
        fprintf(stderr, "Error: peso máximo (%d) fuera del rango [%d, %d]\n", max_weight, MIN_WEIGHT, MAX_WEIGHT);
        return -1;
    }

    if(min_edges < MIN_EDGES_PER_NODE || max_edges > MAX_EDGES_PER_NODE || min_edges > max_edges) {
        fprintf(stderr, "Error: rango de aristas por nodo inválido [%d, %d]\n", MIN_EDGES_PER_NODE, MAX_EDGES_PER_NODE);
        return -1;
    }

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0){
        printf("Parámetros de ejecución:\n");
        printf("Nodos: %d\n", nodes);
        printf("Peso máximo: %d\n", max_weight);
        printf("Aristas mínimas por nodo: %d\n", min_edges);
        printf("Aristas máximas por nodo: %d\n\n", max_edges);
    
        generarGrafo(nodes, max_weight, min_edges, max_edges);
        printf("El proceso %d ha generado el grafo, lo transmite a los demás procesos.\n\n", rank);
    }

    MPI_Bcast(graph, MAX_NODES * sizeof(Node), MPI_BYTE, 0, MPI_COMM_WORLD);

    calcularEstadisticas(nodes, rank, size);

    MPI_Finalize();

    return 0;
}