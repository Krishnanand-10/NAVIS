package com.sih26168.drnav

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler

/**
 * Accelerometer, gyroscope and barometer capture.
 *
 * Two details here are easy to get wrong and expensive to debug later.
 *
 * **Pairing.** Accelerometer and gyroscope events arrive on independent
 * schedules; there is no combined callback. We emit one IMU row per
 * accelerometer event carrying the most recent gyroscope sample, which bounds
 * the pairing error by one gyro interval (5 ms at 200 Hz). That is well under
 * the errors the filter is fighting, but it is an approximation, and it is the
 * reason both raw streams are worth keeping if you later want to interpolate.
 *
 * **Timestamps.** [SensorEvent.timestamp] is documented as nanoseconds since
 * boot, but several OEMs have shipped devices where the base differs from
 * [android.os.SystemClock.elapsedRealtimeNanos] used by the location stack.
 * Everything is stored relative to a single t0 captured at recording start, and
 * the GNSS side uses its own elapsed-realtime stamps, so a constant offset
 * between the two clocks would show up as a fixed lag. Check this on every new
 * phone model before trusting a log: drive a straight line, and look for a
 * consistent along-track bias between the inertial and GNSS solutions.
 */
class ImuCollector(
    context: Context,
    private val rateHz: Int,
    private val onImu: (tNs: Long, accel: FloatArray, gyro: FloatArray) -> Unit,
    private val onBaro: (tNs: Long, hPa: Float) -> Unit,
) : SensorEventListener {

    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accel = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyro = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val pressure = sm.getDefaultSensor(Sensor.TYPE_PRESSURE)

    private val lastGyro = FloatArray(3)
    private var haveGyro = false

    val available: Boolean get() = accel != null && gyro != null
    val hasBarometer: Boolean get() = pressure != null

    fun start(handler: Handler) {
        val periodUs = 1_000_000 / rateHz
        accel?.let { sm.registerListener(this, it, periodUs, handler) }
        gyro?.let { sm.registerListener(this, it, periodUs, handler) }
        // The barometer is slow and only needed for floor level; 5 Hz is plenty.
        pressure?.let { sm.registerListener(this, it, 200_000, handler) }
    }

    fun stop() = sm.unregisterListener(this)

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                System.arraycopy(event.values, 0, lastGyro, 0, 3)
                haveGyro = true
            }
            Sensor.TYPE_ACCELEROMETER ->
                if (haveGyro) onImu(event.timestamp, event.values, lastGyro)
            Sensor.TYPE_PRESSURE -> onBaro(event.timestamp, event.values[0])
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
}
