The Pell-Kahan Cosmic Dynamo: Computational Torque Model for Sgr A*

Este repositório contém a implementação numérica e o framework teórico do Modelo de Torque Computacional Pell-Kahan. A investigação propõe que a gravidade em torno de buracos negros supermassivos, especificamente o Sagittarius A* (Sgr A*), emerge do processamento de geometrias irracionais num espaço-tempo discreto.
🔬 Visão Geral

Ao contrário da visão clássica do espaço-tempo como um palco passivo, este modelo sugere que o horizonte de eventos funciona como um motor de processamento de informação de precisão finita.
Pilares do Modelo:

    Sequência de Pell: Define a geometria da malha discreta do espaço-tempo.

    Algoritmo de Kahan: Modela como o hardware universal gere os resíduos de arredondamento de números irracionais (2​,π,δS​).

    Torque Computacional: Demonstra que a massa de 4.15×106M⊙​ é o trabalho acumulado (Wc​) necessário para sustentar a métrica local.

🚀 Implementação Numérica (pell_kahan_motor.py)

O código Python incluído realiza a simulação do "motor" e gera as evidências estatísticas apresentadas no artigo:

    Simulação de Ciclo de Clock: Modela o drift temporal de 0.51s/ano como latência de processamento.

    Análise de Flares: Identifica harmónicos de Pell na periodicidade das emissões de raios-X.

    Geração de Figuras: Produz automaticamente os 7 gráficos científicos utilizados no manuscrito.

Como Executar:
Bash

git clone https://github.com/stefano-research/pell-kahan-dynamics
cd pell-kahan-dynamics
pip install -r requirements.txt
python pell_kahan_motor.py

📊 Resultados Principais

    Constante η: Derivada como 4.15×10−6, representando a eficiência de conversão informação-métrica.

    Holografia: Conexão direta entre a entropia de Bekenstein-Hawking e o erro residual de Kahan.

    Predição: O modelo prevê perturbações orbitais específicas para a estrela S2, testáveis pela próxima geração de telescópios.

📄 Citação

Se utilizares este modelo ou código na tua investigação, por favor cita:
Snippet de código

### BibTeX
```bibtex
@article{berioni2026pellkahan,
  title={A Computational Torque Model for Timing Anomalies in Sagittarius A*},
  author={Berioni, Stefano},
  journal={Zenodo Archive / arXiv preprint},
  year={2026},
  doi={10.5281/zenodo.18601608},
  url={[https://doi.org/10.5281/zenodo.18601608](https://doi.org/10.5281/zenodo.18601608)}
}
