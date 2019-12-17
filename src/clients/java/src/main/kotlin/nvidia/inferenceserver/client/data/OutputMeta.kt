package nvidia.inferenceserver.client.data

import kotlin.reflect.KClass

data class OutputMeta(val name: String, val dims: List<Long>, val type: KClass<out Number>)