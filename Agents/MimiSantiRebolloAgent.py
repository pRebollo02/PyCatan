import random

from Classes.Constants import *
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Interfaces.AgentInterface import AgentInterface

# --- Pedido a ChatGPT ---
# Algoritmo Genético (AG) paso a paso:
# 1. Representación (Cromosoma): Vector de genes (floats) que codifican pesos para cada decisión del agente.
# 2. Población inicial: Conjunto de cromosomas aleatorios.
# 3. Evaluación (Función de fitness): Para cada cromosoma,
#    a) Crear agente con esos pesos.
#    b) Jugar N partidas y medir 'gain' = tasa de victorias ponderada por puntos medios.
#    c) Fitness = gain (maximizar). El 'gain' representa la ganancia esperada en recursos/puntos.
# 4. Selección: Elegir padres según fitness (torneo o ruleta).
# 5. Cruce: Mezclar genes de padres (un punto o uniforme).
# 6. Mutación: Alterar genes con ruido gaussian o probabilidad baja.
# 7. Reemplazo: Mantener mejores (elitismo) + nuevos hijos.
# 8. Repetir pasos 3-7 hasta criterio de parada (generaciones).
# 9. Solución: Mejor cromosoma final.
#
# 'gain' en on_trade_offer y on_commerce_phase:
# - gain = recursos recibidos (sum(reci))
# - cost = recursos entregados (sum(gives))
# - Se acepta si weight_trade * gain >= cost, ajustando la aversión al riesgo/comercio.


class MimiSantiRebolloAgent(AgentInterface):
    """ esto está pedido a chatgpt
    Agent parametrizado por un cromosoma que controla múltiples decisiones:
      - weight_init_pos: importancia de probabilidad de recursos en colocación inicial
      - weight_block: prioridad para bloquear rivales al mover ladrón
      - weight_discard: umbral para descartes cuando >7 cartas
      - weight_trade: preferencia en oferta de comercio
      - weight_build_road, weight_build_town, weight_build_city: umbrales de construcción
      - Otros genes opcionales para uso de cartas de desarrollo
    El cromosoma se pasa como lista de floats.
    """

    def __init__(self, agent_id, chromosome):
        super().__init__(agent_id)
        assert len(chromosome) >= 8, "Cromosoma debe tener al menos 8 genes"

        # Genes básicos
        self.weight_init_pos    = chromosome[0]
        self.weight_block       = chromosome[1]
        self.weight_discard     = chromosome[2]
        self.weight_trade       = chromosome[3]
        self.weight_build_road  = chromosome[4]
        self.weight_build_town  = chromosome[5]
        self.weight_build_city  = chromosome[6]
        # Peso para jugar año de la cosecha vs construir carreteras
        self.weight_dev_card    = chromosome[7]
        # Genes adicionales opcionales
        self.extra_genes        = chromosome[8:]

        # Estado interno heredado de AdrianHerasAgent
        self.town_number = 0
        self.material_given_more_than_three = None
        self.year_of_plenty_material_one = MaterialConstants.CEREAL
        self.year_of_plenty_material_two = MaterialConstants.MINERAL

    def on_game_start(self, board_instance):
        # Decidir nodo inicial ponderado genéticamente
        self.board = board_instance
        possibilities = self.board.valid_starting_nodes()
        best_node = None
        best_score = float('-inf')
        for node_id in possibilities:
            score = self.weight_init_pos * sum(
                1 for tid in self.board.nodes[node_id]['contacting_terrain']
                if self.board.terrain[tid]['probability'] in (6, 8)
            )
            if score > best_score:
                best_score = score
                best_node = node_id

        # Seleccionar carretera aleatoria para simplicidad
        roads = self.board.nodes[best_node]['adjacent']
        road_to = random.choice(roads) if roads else roads[0]
        self.town_number += 1
        return best_node, road_to

    def on_trade_offer(self, board_instance, offer=TradeOffer(), player_id=int):
        # Aceptar oferta si ganancia ponderada supera umbral
        gain = sum(offer.gives.to_list())
        cost = sum(offer.receives.to_list())
        return self.weight_trade * gain >= cost

    def on_having_more_than_7_materials_when_thief_is_called(self):
        # Descartes heurísticos
        while self.hand.get_total() > 7:
            # descartar según peso_discard y valor material
            # Simplificación: descartar cereal primero
            if self.hand.resources.cereal > 0:
                self.hand.remove_material(MaterialConstants.CEREAL,1)
            else:
                mat = random.choice(MaterialConstants.ALL)
                if self.hand.resources.get_from_id(mat) > 0:
                    self.hand.remove_material(mat,1)
        return self.hand

    def on_turn_start(self):
        # Si tiene mano de cartas de desarrollo
        knight_cards = self.development_cards_hand.find_card_by_effect(DevelopmentCardConstants.KNIGHT_EFFECT)
        return knight_cards[0] if len(knight_cards) > 0  else None

    def on_moving_thief(self):
        moves = self.board.valid_thief_moves()
        best_move = None
        best_score = float('-inf')
        for hex_idx in moves:
            blocked = self.board.count_pieces_on_hex(hex_idx)
            score = self.weight_block * blocked
            if score > best_score:
                best_score = score
                best_move = hex_idx
        return {'terrain': best_move, 'player': -1}

    def on_turn_end(self):
        return super().on_turn_end()

    def on_commerce_phase(self):
        """
        Negociación parametrizada genéticamente:
          - Jugar carta de monopolio si gain ponderado supera threshold
          - Ofrecer/comprar materiales según weight_trade
        """
        # 1. Jugar carta de monopolio si corresponde
        if self.material_given_more_than_three is not None:
            for i, card in enumerate(self.development_cards_hand.hand):
                if card.effect == DevelopmentCardConstants.MONOPOLY_EFFECT:
                    # jugar monopolio si ganancia esperada ponderada alta
                    return self.development_cards_hand.select_card(i)

        # 2. Calcular déficit y surplus
        # Definir necesidades para ciudad o pueblo
        needs = None
        if self.town_number >= 1:
            needs = BuildConstants.CITY
        else:
            needs = Materials(1,0,1,1,1)

        # 3. Si puede construir no comercia
        if self.hand.resources.has_more(needs):
            self.material_given_more_than_three = None
            return None

        # 4. Determinar lo que pide y lo que ofrece basado en necesidad
        gives = Materials(0,0,0,0,0)
        receives = Materials(0,0,0,0,0)
        # calcular deficit: lo que le falta de la construcción objetivo
        if isinstance(needs, Materials):
            deficit = [max(0, req - self.hand.resources.get_from_id(mid))
                       for mid, req in enumerate(needs.to_list())]
        else:
            # para CITY, cereal=2, mineral=3
            deficit = [2 - self.hand.resources.cereal, 3 - self.hand.resources.mineral, 0,0,0]
        # solicita según deficit
        for mat_id, amount in enumerate(deficit):
            if amount > 0:
                # agregar manualmente al objeto receives
                current = receives.to_list()
                current[mat_id] += amount
                receives = Materials(current[0], current[1], current[2], current[3], current[4])
        # surplus: todo lo que sobra
        for mat_id in range(5):
            have = self.hand.resources.get_from_id(mat_id)
            needed = deficit[mat_id]
            if have > needed:
                extra = have - needed
                # sumar manualmente al objeto gives
                current = gives.to_list()
                current[mat_id] += extra
                gives = Materials(current[0], current[1], current[2], current[3], current[4])

        # 5. Decidir con peso_trade si realizar oferta. Decidir con peso_trade si realizar oferta
        total_gain = sum(receives.to_list())
        total_cost = sum(gives.to_list())
        if self.weight_trade * total_gain >= total_cost:
            offer = TradeOffer(gives, receives)
            self.material_given_more_than_three = max(receives.to_list())
            return offer

        # 6. Si no conviene retorna None
        return None

    def on_build_phase(self, board_instance):
        self.board = board_instance
        # Intentar jugar carta de desarrollo si peso lo sugiere
        for i, card in enumerate(self.development_cards_hand.hand):
            if random.random() < self.weight_dev_card:
                return self.development_cards_hand.select_card(i)

        # Construcción ponderada
        actions = self.board.valid_build_actions(self.id)
        best_act = None
        best_score = float('-inf')
        for act in actions:
            if act['building'] == BuildConstants.ROAD:
                score = self.weight_build_road
            elif act['building'] == BuildConstants.TOWN:
                score = self.weight_build_town
            elif act['building'] == BuildConstants.CITY:
                score = self.weight_build_city
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_act = act
        return best_act

    def on_monopoly_card_use(self):
        # Elige el material que más haya intercambiado (variable global de esta clase)
        return self.material_given_more_than_three

    def on_road_building_card_use(self):
        return super().on_road_building_card_use()

    def on_year_of_plenty_card_use(self):
        return super().on_year_of_plenty_card_use()

