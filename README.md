# analytics-catalogue
The Diastema Analytics Catalogue

## How to Use
This project is inteded to by used by [the Diastema orchestrator](https://github.com/DIASTEMA-UPRC/orchestrator). If you need information on how to use it alongside other components, please refer to that documentation. If you need information on how to run this in isolation, follow the steps described below:

### Prerequisites
+ Docker
+ MinIO

### How to Build
```bash
docker build -t analytics-catalogue:latest . -f Dockerfile.dev
```

### How to Run
The dev image mounts the src folder as a volume under `/app/src`

#### How to Use Image
```bash
docker run --rm -d analytics-catalogue
docker exec -it <container-name-or-id> bash
```

#### How to Execute a Spark Job
```bash
python src/<JobType> <algorithm> <input_path> <output_path> <target_column>
```

##### Spark Job Arguments

| Argument | Description | Example |
| -------- | ----------- | ------- |
| JobType | The type of job to run | ClassificationJob.py |
| algorithm | The type of algorithm to run | decisiontree |
| input_path | The MinIO path to look for input data | /in/ex1/data |
| output_path | The MinIO path to output resulting data | /out/ex1/data |

## License
Licensed under the [Apache License Version 2.0](README) by [Konstantinos Voulgaris](https://github.com/konvoulgaris) for the research project Diastema
